import cv2

image = cv2.imread('./image.jpg', cv2.IMREAD_COLOR)

a = 300
b = 300

zoom = cv2.resize(image, (a,b), interpolation=cv2.INTER_AREA)


