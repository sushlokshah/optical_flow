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
                            minDistance = 7,
                            blockSize = 7 )
        
        features1 = cv.goodFeaturesToTrack(image1_gray, mask = None, **feature_params)
        features2, mask, err = cv.calcOpticalFlowPyrLK(image1_gray, image2_gray, features1, None, **lk_params)
        
        # Select good points
        if features2 is not None:
            good_new = features2[mask==1]
            good_old = features1[mask==1]
        
        flow  = good_new -good_old
        mag, ang = cv.cartToPolar(flow[:,0], flow[:,1])
        mag_map = np.zeros(image1_gray.shape).astype(np.float32)
        ang_map = np.zeros(image1_gray.shape).astype(np.float32)
        for i in range(len(flow)):
            # print(good_old[i][1])
            # print(good_old)
            mag_map[int(np.floor(good_old[i,1])),int(np.floor(good_old[i,0]))] = mag[i]
            ang_map[int(np.floor(good_old[i,1])),int(np.floor(good_old[i,0]))] = ang[i]

        # plt.imshow(mag_map)
        # plt.show()
        # print(mag,ang)
        hsv = np.zeros_like(image1)
        hsv[:,:,1] = 255
        hsv[:,:,0] = ang_map*180/np.pi/2
        hsv[:,:,2] = cv.normalize(mag_map, None, 0, 255, cv.NORM_MINMAX)
        # # print(flow)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
        return bgr
        

    elif method == "Fast_features":
        fast = cv.FastFeatureDetector_create()
        fast.setNonmaxSuppression(1)
        fast.setThreshold(30)
        kp = fast.detect(image1_gray,None)
        features1 = np.float32([ kp[m].pt for m in range(len(kp))]).reshape(-1,1,2)
        features2, mask, err = cv.calcOpticalFlowPyrLK(image1_gray, image2_gray, features1, None, **lk_params)
        
        # Select good points
        if features2 is not None:
            good_new = features2[mask==1]
            good_old = features1[mask==1]
        
        flow  = good_new -good_old
        mag, ang = cv.cartToPolar(flow[:,0], flow[:,1])
        mag_map = np.zeros(image1_gray.shape).astype(np.float32)
        ang_map = np.zeros(image1_gray.shape).astype(np.float32)
        for i in range(len(flow)):
            # print(good_old[i][1])
            # print(good_old)
            mag_map[int(np.floor(good_old[i,1])),int(np.floor(good_old[i,0]))] = mag[i]
            ang_map[int(np.floor(good_old[i,1])),int(np.floor(good_old[i,0]))] = ang[i]

        # plt.imshow(mag_map)
        # plt.show()
        # print(mag,ang)
        hsv = np.zeros_like(image1)
        hsv[:,:,1] = 255
        hsv[:,:,0] = ang_map*180/np.pi/2
        hsv[:,:,2] = cv.normalize(mag_map, None, 0, 255, cv.NORM_MINMAX)
        # # print(flow)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
        return bgr

def Farneback_flow(image1,image2):
    prvs = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros_like(image1)
    hsv[..., 1] = 255
    next = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    print(flow[..., 0].shape)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    print(mag.shape,ang.shape)
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr