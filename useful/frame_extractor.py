import cv2
import time
import sys
vidcap = cv2.VideoCapture(0)
success,image = vidcap.read()
count = 0
success = True
t = float(sys.argv[1])
while success:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1
  time.sleep(t)
