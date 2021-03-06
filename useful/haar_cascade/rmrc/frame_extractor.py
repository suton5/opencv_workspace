import cv2
import time
import os
import sys

vidcap = cv2.VideoCapture(0)
success,image = vidcap.read()
count = 59
success = True
t = float(sys.argv[1])

while success:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  image = cv2.resize(image, (100, 100))
  cv2.imwrite("negatives/"+str(count)+".jpg", image)     # save frame as JPEG file
  count += 1
  time.sleep(t)
