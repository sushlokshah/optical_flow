import numpy as np
import cv2 as cv 
import os 
import argparse
import matplotlib.pyplot as plt

def lucus_kanade_flow(image1,image2,method = None):
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    image1_gray = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    image2_gray = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    
    if method == "GoodFeaturesToTrack":
        # params for ShiTomasi corner detection
        feature_params = dict( maxCorners = 10000,
                            qualityLevel = 0.3,
                            minDistance = 1,
                            blockSize = 3 )
        
        features1 = cv.goodFeaturesToTrack(image1_gray, mask = None, **feature_params)
        features2, mask, err = cv.calcOpticalFlowPyrLK(image1_gray, image2_gray, features1, None, **lk_params)
        
        # Select good points
        if features2 is not None:
            good_new = features2[mask==1]
            good_old = features1[mask==1]
        
        flow  = good_new -good_old
        # mag, ang = cv.cartToPolar(flow[:,0], flow[:,1])
        u_map = np.zeros(image1_gray.shape).astype(np.float32)
        v_map = np.zeros(image1_gray.shape).astype(np.float32)
        mask = np.zeros(image1_gray.shape).astype(np.float32)
        for i in range(len(flow)):
            # print(good_old[i][1])
            # print(good_old)
            u_map[int(np.floor(good_old[i,1])),int(np.floor(good_old[i,0]))] = flow[i,0]
            v_map[int(np.floor(good_old[i,1])),int(np.floor(good_old[i,0]))] = flow[i,1]
            mask[int(np.floor(good_old[i,1])),int(np.floor(good_old[i,0]))] = 1

        # plt.imshow(mag_map)
        # plt.show()
        # print(mag,ang)
        # hsv = np.zeros_like(image1)
        # hsv[:,:,1] = 255
        # hsv[:,:,0] = ang_map*180/np.pi/2
        # hsv[:,:,2] = cv.normalize(mag_map, None, 0, 255, cv.NORM_MINMAX)
        # # print(flow)
        # bgr = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
        # mask = np.zeros_like(image1_gray)
        flow_format = np.zeros([image1_gray.shape[0],image1_gray.shape[1],2])
        flow_format[:,:,0] = u_map
        flow_format[:,:,1] = v_map
        return flow_format,mask
        

    elif method == "Fast_features":
        fast = cv.FastFeatureDetector_create()
        fast.setNonmaxSuppression(1)
        fast.setThreshold(10)
        kp = fast.detect(image1_gray,None)
        features1 = np.float32([ kp[m].pt for m in range(len(kp))]).reshape(-1,1,2)
        features2, mask, err = cv.calcOpticalFlowPyrLK(image1_gray, image2_gray, features1, None, **lk_params)
        
        # Select good points
        if features2 is not None:
            good_new = features2[mask==1]
            good_old = features1[mask==1]
        
        flow  = good_new -good_old
        # mag, ang = cv.cartToPolar(flow[:,0], flow[:,1])
        u_map = np.zeros(image1_gray.shape).astype(np.float32)
        v_map = np.zeros(image1_gray.shape).astype(np.float32)
        mask = np.zeros(image1_gray.shape).astype(np.float32)
        for i in range(len(flow)):
            # print(good_old[i][1])
            # print(good_old)
            u_map[int(np.floor(good_old[i,1])),int(np.floor(good_old[i,0]))] = flow[i,0]
            v_map[int(np.floor(good_old[i,1])),int(np.floor(good_old[i,0]))] = flow[i,1]
            mask[int(np.floor(good_old[i,1])),int(np.floor(good_old[i,0]))] = 1

        # plt.imshow(mag_map)
        # plt.show()
        # print(mag,ang)
        # hsv = np.zeros_like(image1)
        # hsv[:,:,1] = 255
        # hsv[:,:,0] = ang_map*180/np.pi/2
        # hsv[:,:,2] = cv.normalize(mag_map, None, 0, 255, cv.NORM_MINMAX)
        # # print(flow)
        # bgr = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
        # mask = np.zeros_like(image1_gray)
        flow_format = np.zeros([image1_gray.shape[0],image1_gray.shape[1],2])
        flow_format[:,:,0] = u_map
        flow_format[:,:,1] = v_map
        return flow_format,mask

def Farneback_flow(image1,image2):
    prvs = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    # hsv = np.zeros_like(image1)
    # hsv[..., 1] = 255
    next = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # print(flow.shape)
    mask = np.zeros_like(prvs).astype(np.float32)
    mask[:,:] = 1
    # mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    # print(mag.shape,ang.shape)
    # hsv[..., 0] = ang*180/np.pi/2
    # hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    # bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return flow,mask

if __name__ == "__main__":
    image1 = cv.imread("/home/sushlok/optical_flow/dataset/data_scene_flow/training/image_2/000000_10.png",1)
    image2 = cv.imread("/home/sushlok/optical_flow/dataset/data_scene_flow/training/image_2/000000_11.png",1)
    Farneback_flow(image1,image2)