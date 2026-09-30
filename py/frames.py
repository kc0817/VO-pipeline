import sys

import cv2

path = "/mnt/c/Users/CHENY1/Downloads/nemo-data-straight-down-acre-trimmed-1.mp4"
fps = 30
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    print("failed to open video capture; exiting")
    sys.exit(0)

while True:
    success, frame = cap.read()
    if not success:
        print("failed video capture read; exiting")
        break
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("images", gray_scale)
    if cv2.waitKey(1000 // fps) != -1:
        print("user input detected; stopping")
        break

cap.release()
cv2.destroyAllWindows()
