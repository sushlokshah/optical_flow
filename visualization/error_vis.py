import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt 
import os
path = "results/flownet2"

def flow_vis(path,i):
    # print(path)
    flow = cv.imread(path+ "/result_flow_img/" + i , cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    kitti_flow_error = cv.imread(path+ "/errors_flow_img/" + i , cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH) 
    EPE_map = cv.imread(path+ "/error_vis/" + i , cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH) 
    img = cv.imread(path+ "/image_0/" + i , cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    EPE_map = cv.cvtColor(EPE_map, cv.COLOR_BGR2RGB)
    kitti_flow_error = cv.cvtColor(kitti_flow_error, cv.COLOR_BGR2RGB)
    flow  = cv.cvtColor(flow, cv.COLOR_BGR2RGB)
    EPE_map = cv.addWeighted(img, 0.2, EPE_map, 0.8, 0)
    kitti_flow_error = cv.addWeighted(img, 0.2, kitti_flow_error, 0.8, 0)
    flow = cv.addWeighted(img, 0.2, flow, 0.8, 0)
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(img)
    axs[0, 0].set_title('image')
    axs[0, 1].imshow(EPE_map)
    axs[0, 1].set_title('EPE_map')
    axs[1, 0].imshow(kitti_flow_error)
    axs[1, 0].set_title('kitti_flow_error')
    axs[1, 1].imshow(flow)
    axs[1, 1].set_title('flow')
    
    plt.show()
    
    
list_img = sorted(os.listdir(path))
print(list_img)
for i in list_img:
    if(i[-4:] == ".png"):
        print(i)
        flow_vis(path,i)