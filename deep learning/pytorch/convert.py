def attributes_opencv(track):
    x1,y1,w,h = track.tlwh.astype('int16')
    track_id = track.track_id
    x = int(x1 + w/2)
    y = int(y1 + h/2)
    x2 = int(x1 + w)
    y2 = int(y1 + h)
    return dict(xyxy = [x1,y1,x2,y2], xywh = [x,y,w,h], id = track_id)

def convert_xyxy(xywh):
    x1,y1,w,h = xywh.astype('int16')
    x2 = int(x1 + w)
    y2 = int(y1 + h)
    return [x1,y1,x2,y2]

def convert_xywh(xywh):
    x1,y1,w,h = xywh.astype('int16')
    x = int(x1 + w/2)
    y = int(y1 + h/2)
    return [x,y,w,h]

def convert_bytetrack(detections):
    # Remove unwanted detections
    converted = []
    for det in detections:
        det.pop('xywh')
        converted.append([det.get('xyxy')[0],det.get('xyxy')[1],det.get('xyxy')[2],det.get('xyxy')[3], det.get('conf'),det.get('cls')])
    return converted