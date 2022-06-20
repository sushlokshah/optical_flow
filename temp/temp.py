import cv2 as cv
import numpy as np

img1 = cv.imread('/home/sushlok/new_approach/datasets/vkitti/vkitti_1.3.1_flowgt/0001/fog/00000.png',cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
img2 = cv.imread('/home/sushlok/optical_flow/GMFlowNet/result/vkitti/000000.png',cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
print(img1[:,:,0])