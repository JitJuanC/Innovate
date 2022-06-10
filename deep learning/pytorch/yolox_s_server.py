import flask
import argparse
import os
import time
from loguru import logger
import numpy as np
import cv2
import torch
import json

from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import fuse_model, get_model_info, postprocess, vis


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# Add all the default value first as we don't use command-line arguments
def make_parser():
    parser = argparse.ArgumentParser("YOLOX Integration by Flask!")
    parser.add_argument(
        "-demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default="yolox-s", help="model name")

    parser.add_argument(
        "--path", default="", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        # default="/home/ubuntu/YOLOX_MOT/YOLOX/exps/example/yolox_voc/yolox_voc_s.py",
        default="/workspace/YOLOX/exps/example/yolox_voc/yolox_voc_s.py",
        #default="/home/ubuntu/TEST/YOLOX/exps/example/yolox_voc/yolox_voc_s.py",
        type=str,
        help="pls input your experiment description file",
    )
    # parser.add_argument("-c", "--ckpt", default="/home/ubuntu/YOLOX_MOT/YOLOX/YOLOX_pretrained_weights/yolox_s.pth", type=str, help="ckpt for eval")
    parser.add_argument("-c", "--ckpt", default="/workspace/models/yolox_s.pth", type=str, help="ckpt for eval")
    # parser.add_argument("-c", "--ckpt", default="/home/ubuntu/YOLOX_old/YOLOX_outputs/yolox_voc_s/ANPRv3___latest_ckpt.pth", type=str, help="ckpt for eval")
    # parser.add_argument("-c", "--ckpt", default="/home/ubuntu/YOLOX_Sam/YOLOX_outputs/yolox_voc_s/latest_ckpt.pth", type=str, help="ckpt for eval")
    # ByteTrack MOT weights
    # parser.add_argument("-c", "--ckpt", default="/home/ubuntu/ByteTrack/pretrained/bytetrack_s_mot17.pth.tar", type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.25, type=float, help="test conf")
    parser.add_argument("--nms", default=0.45, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser.parse_args()


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        # Convert all torch tensors to numpy
        bboxes = bboxes.numpy()
        scores = scores.numpy()
        cls = cls.numpy()

        # Extract them one data at a time
        raw_res = [bboxes, scores, cls, self.cls_names]
        # print(raw_res)
        # print(len(raw_res))
        # print(bboxes, scores, cls, cls_conf, self.cls_names)
        # print(type(bboxes), type(scores), type(cls), type(cls_conf), type(self.cls_names))
        return raw_res

        # vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        # return vis_res


# argument needed --> predictor, path
def image_demo(predictor, frame):
    outputs, img_info = predictor.inference(frame)
    result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
    # print(f"Outputs: {outputs}", outputs[0])
    # print(f"Outputs: {outputs}", type(outputs))
    # print(f"IMG INFO: {img_info}", type(img_info))
    # print(f"Results: {type(result_image[0])}, {type(result_image[1])}, {type(result_image[2])}, {type(result_image[3])}")
    # print(f"Results: {result_image}")
    return result_image

def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame) 
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break

def main(args):
    exp = get_exp(args.exp_file, args.name)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(exp.num_classes)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~")

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # file_name = os.path.join(exp.output_dir, args.experiment_name)
    # os.makedirs(file_name, exist_ok=True)

    # vis_folder = None
    # if args.save_result:
    #     vis_folder = os.path.join(file_name, "vis_res")
    #     os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)
        #exp.test_size = [args.tsize, args.tsize]

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        # if args.ckpt is None:
        #     ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        # else:
        #     ckpt_file = args.ckpt
        ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    # if args.trt:
    #     assert not args.fuse, "TensorRT model is not support model fusing!"
    #     trt_file = os.path.join(file_name, "model_trt.pth")
    #     assert os.path.exists(
    #         trt_file
    #     ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
    #     model.head.decode_in_inference = False
    #     decoder = model.head.decode_outputs
    #     logger.info("Using TensorRT to inference")
    # else:
    trt_file = None
    decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )

    return predictor
    # current_time = time.localtime()
    # # Open app.route here, use cv2.imencode to get image from requests
    # if args.demo == "image":
    #     image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    # elif args.demo == "video" or args.demo == "webcam":
    #     imageflow_demo(predictor, vis_folder, current_time, args)

def format_dets(raw_dets, cls, thres):
    dets2 = []
    # Skip some nonsense frame
    if len(raw_dets) > 4:
        return dets2
    boxes, scores, cls_ids, class_names = raw_dets
    boxes = boxes.astype('int16').tolist()
    cls_ids = cls_ids.astype('int16').tolist()
    scores = scores.astype('float32').tolist()
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        if scores[i] < thres:
            continue
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])

        w = int(x2 - x1)
        x = int(x1 + w / 2)
        h = int(y2 - y1)
        y = int(y1 + w / 2)

        det = {}
        det['xyxy'] = [int(x1),int(y1),int(x2),int(y2)]
        det['xywh'] = [x,y,w,h]
        det['conf'] = scores[i]
        det['cls'] = class_names[cls_id]
        if det['cls'] in cls or cls is None:
            dets2.append(det)

    return dets2


if __name__ == "__main__":
    args = make_parser()

    # initialize Flask
    app = flask.Flask(__name__)
    app.config['DEBUG'] = True

    predictor = main(args)

    # receive frame in numpy from client to process
    @app.route('/predict', methods=['POST'])
    def predict():
        # imencode (to numpy) then change to bytes
        received = flask.request.get_data()
        details = json.loads(flask.request.get_json())
        im_np = np.frombuffer(received, dtype='uint8')
        # decode bytes to frame
        frame = cv2.imdecode(im_np, cv2.IMREAD_COLOR)
        # send frame to be inferenced
        results = image_demo(predictor, frame)
        conf = predictor.confthre
        print("predictor.confthre: ", predictor.confthre)
        # dets = format_dets(results, 0.25)
        dets = format_dets(results, details['class'], details['confidence'])
        final = json.dumps(dets)
        return final

    app.run(port = 5002)
