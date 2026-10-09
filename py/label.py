import cv2

img_num = 745

read_path = "../data/frame" + str(img_num) + ".png"
output_path = "../flow_fields/label" + str(img_num) + ".png"

p1 = (700, 800)
p2 = (p1[0] + 30, p1[1] + 30)
color = (0, 0, 255)
thickness = 2

img = cv2.imread(read_path)
if img is not None:
    cv2.rectangle(img, p1, p2, color, thickness)

    _ = cv2.imwrite(output_path, img)
    print("successfully written to " + output_path)
