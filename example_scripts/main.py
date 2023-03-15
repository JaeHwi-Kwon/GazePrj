import mediapipe as mp
import numpy as np
import cv2 as cv

frame = cv.imread('elon_musk.jpg')

while True:
    cv.imshow('Main',frame)
    key = cv.waitKey(1)
    if( key == ord('q')):
        break

frame.release()
cv.destroyAllWindows()