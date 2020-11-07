#!/usr/bin/env python
import cv2
from lib.display import Display

width = 3840 // 4
height = 2160 // 4

display = Display(width, height)
capture = cv2.VideoCapture("./video/production_ID_4608279.mp4")

while capture.isOpened():
    ret, frame = capture.read()
    if ret:
        display.draw(frame)
    else:
        break
