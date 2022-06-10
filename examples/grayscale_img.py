import os
import cv2

directory = "/home/ubuntu/tata/"
target = "tigor"
full_path = os.path.join(directory, target)

for img in os.listdir(full_path):
    new_img = cv2.imread(os.path.join(full_path, img))
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(full_path, img), gray)