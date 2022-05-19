import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
image1 = cv.imread("/home/sushlok/optical_flow/dataset/data_scene_flow/training/image_2/000000_10.png",1)
image2 = cv.imread("/home/sushlok/optical_flow/dataset/data_scene_flow/training/image_2/000000_11.png",1)
prvs = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(image1)
hsv[..., 1] = 255
next = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
print(flow[..., 0].shape)
mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 0] = ang*180/np.pi/2
hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
plt.imshow(bgr)
plt.show()