import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img1 = cv.imread('/home/sushlok/optical_flow/GMFlowNet/result/vkitti_flow2/00033.png',cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)

flow_est = img1[:,:,1:]*(1/(2**16 - 1))
flow_est = flow_est*2 - 1
flow_est[:,:,0] = flow_est[:,:,0]*(flow_est.shape[1]-1)
flow_est[:,:,1] = flow_est[:,:,1]*(flow_est.shape[0]-1)
# print(flow_est.min(),flow_est.max())


# read npz file
flow = np.load('/home/sushlok/carla-python/flow_npz/1148_299.25492858454527.npz')
flow = np.array(flow["arr_0"]).reshape((flow_est.shape[0],flow_est.shape[1],2))

# print(flow)
img2 = cv.imread('/home/sushlok/carla-python/flow/0001/flow2/1148_299.25492858454527.png',cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2HSV)
ang = img2[:,:,0]/28.64788975654116
# print(ang)
mag = img2[:,:,2]/2550.0
# print(mag)
u = mag*np.sin(ang)*(flow_est.shape[0]-1)
v = mag*np.cos(ang)*(flow_est.shape[1]-1)
flow_gt = np.stack((u,v),axis=2)
# print(v.max(),v.min())

# est_flow = img1[:,:,1:]*(1/(2**16 - 1))
# print(est_flow.max(),est_flow.min())
# est_flow = est_flow*2 - 1
# print(u.max(),u.min())
fig, ax = plt.subplots(1,3)
ax[0].imshow(flow_est[:,:,0])
ax[1].imshow(u)
ax[2].imshow(abs(flow_est[:,:,0] - u))
print(np.average(abs(flow_est[:,:,0] - u)))
# ax[0].imshow(flow_est[:,:,1])
# ax[1].imshow(v)
# ax[2].imshow((-1*flow[:,:,0]))
plt.show()
print("finally")