import sys

import cv2

show_imgs = False
if len(sys.argv) == 2:
    show_imgs = True

path = "/mnt/c/Users/CHENY1/Downloads/nemo-data-straight-down-acre-trimmed-1.mp4"
fps = 30
cap = cv2.VideoCapture(path)
if not cap.isOpened():
    print("failed to open video capture; exiting")
    sys.exit(0)

n = 0
while True:
    success, frame = cap.read()
    if not success:
        print("failed video capture read at frame " + str(n) + "; exiting")
        break
    if show_imgs:
        cv2.imshow("images", frame)
        if cv2.waitKey(1000 // fps) != -1:
            print("user input detected; stopping")
            break
    write_success = cv2.imwrite("../data/frame" + str(n) + ".png", frame)
    if not write_success:
        print("failed to write img " + str(n) + ".png; exiting")
        break
    if n % 50 == 0:
        print("saved frame" + str(n) + ".png")
    n += 1

cap.release()
cv2.destroyAllWindows()
