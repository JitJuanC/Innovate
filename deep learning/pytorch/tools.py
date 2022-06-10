"""Sam's tools

Extra functions to use in conjunction with the yolox server 

Easy Usage: 
1. Import this Class
2. Create a "Tools" instance
3. Use the tool to get the detections from server
"""

import cv2
from requests import post
import numpy as np
import json

class Tools:
    # cls is type Set
    def __init__(self, target, cls=None, conf=0.1):
        self.target = f'http://127.0.0.1:{target}/predict'
        self.details = {
            "class":cls,
            "confidence":conf}

    def start(self, frame):
        ret, im_np = cv2.imencode('.jpg', frame)
        im_byte = im_np.tobytes()
        dets = self.get_dets(im_byte)
        return dets

    def get_dets(self, im_byte):
        metadata = post(self.target, data=im_byte, json=self.details)
        # print(metadata.text)
        return metadata.json()

    def detection(self, frame):
        dets = self.start(frame)
        # print(f"DETS, {dets}")
        return dets


