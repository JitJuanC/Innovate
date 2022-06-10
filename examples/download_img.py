import cv2
import numpy as np
import requests
import os
import sys

def save(url, extra, full_path, ext='.jpg', start=1, required=False):
    while True:
        if start > 9 and required:
            extra = extra[:-1]
            required = False
        res = requests.get(f"{url}{extra}{start}{ext}", headers={'User-agent': 'Mozilla/5.0'})
        if res.status_code != 200:
            break
        img_np = np.frombuffer(res.content, dtype='uint8')
        img = cv2.imdecode(img_np, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(full_path, f"{target}{extra}{start}.jpg"), img)
        print(os.path.join(full_path, f"{target}{extra}{start}.jpg"))
        start += 1

if __name__=='__main__':
    # Creating directory for dataset
    directory = "/home/ubuntu/tata/"
    target = "safari"
    full_path = os.path.join(directory, target)
    os.makedirs(full_path, exist_ok=True)
    print("Paths good")


    # Getting dataset
    extra = "0"
    url = "https://cars.tatamotors.com/images/safari/gallery/"

    save(url, extra, full_path, ext=".jpg", start=27, required=True)
    