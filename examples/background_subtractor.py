import cv2 as cv
import time
import numpy as np

bgs = cv.createBackgroundSubtractorMOG2(history=100, varThreshold=80, detectShadows=False)
cap = cv.VideoCapture("test_video.mov") 

number = 0
while True:
    ret, frame = cap.read() 
    if frame is None:
        break
    frame = cv.resize(frame, (640,480)) # to make the output smaller in size

    mask = bgs.apply(frame) # do background subtraction on every frame (The difference of pixels)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) # contours is the outer layer of a shape

    # cv.imshow("Video", frame)
    cv.imshow("Mask", mask)
    # print(number)
    # time.sleep(2)
    # number += 1

    key = cv.waitKey(30) # the number determines the milisecond that it fetches each frame in this loop, 0 is wait until any key is pressed

    if key == 27: # 27 is 'esc' key on keyboard
        break

cap.release()
cv.destroyAllWindows()