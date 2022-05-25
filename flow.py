import numpy as np
import cv2 as cv 
import os 
import sys
import argparse
import matplotlib.pyplot as plt
import time
import torch
# import numba

from classical_flow.classical_flow_methods import lucus_kanade_flow, Farneback_flow
from spynet.run import estimate

def data_formating(flow, mask):
    # flow = (2**15)/128 - flow
    flow_kitti_format = np.zeros([flow.shape[0],flow.shape[1],3])
    flow_kitti_format[:,:,1:] = flow
    flow_kitti_format[:,:,0] = mask
    # plt.imshow(flow_kitti_format)
    # plt.show()
    return flow_kitti_format

def optical_flow(image1,image2, method = None):
    if method == "lucas_kanade_GoodFeaturesToTrack":
        flow, mask= lucus_kanade_flow(image1,image2,method="GoodFeaturesToTrack")
    elif method == "lucas_kanade_Fast_features":
        flow, mask = lucus_kanade_flow(image1,image2,method="Fast_features")
    elif method == "Farneback_flow":
        flow, mask = Farneback_flow(image1,image2)
    elif method == "spynet":
        # print(torch.FloatTensor((image1/255).transpose(2, 0, 1)).shape)
        flow = estimate(torch.FloatTensor((image1/255).transpose(2, 0, 1)),torch.FloatTensor((image1/255).transpose(2, 0, 1)))
        flow = flow.numpy().transpose(1, 2, 0)
        mask = np.ones([flow.shape[0],flow.shape[1]])
    return flow, mask

def save_data(path,flow,mask):
    """output dataformat and the folder format

    Args:
        path (string): path and the file name of the images
        flow (matrix): as per the format specified in the kitti benchmark
        
        directory structure
        |-- flow    (Flow fields between first and second image)
            |-- 000000_10.png
            |-- ...
            |-- 000199_10.png
        
        output format:
        Optical flow maps are saved as 3-channel uint16 PNG images: The first channel
        contains the u-component, the second channel the v-component and the third
        channel denotes if the pixel is valid or not (1 if true, 0 otherwise). To convert
        the u-/v-flow into floating point values, convert the value to float, subtract
        2^15 and divide the result by 64.0:

        flow_u(u,v) = ((float)I(u,v,1)-2^15)/64.0;
        flow_v(u,v) = ((float)I(u,v,2)-2^15)/64.0;
        valid(u,v)  = (bool)I(u,v,3);

    """
    flow_kitti_format = data_formating(flow,mask)
    # flow_kitti_format = flow_kitti_format.astype(np.uint16)
    cv.imwrite(path,flow_kitti_format*255)

def evalaute_data(path, method = None):
        os.system("g++ -O3 -DNDEBUG -o ./evalution/cpp/evaluate_scene_flow ./evalution/cpp/evaluate_scene_flow.cpp -lpng")
        os.system("./evalution/cpp/evaluate_scene_flow "+ args["method"])

def calculate_flow(args):
    if not os.path.exists(args["output_dir"] +"/"+ args["method"]):
        os.makedirs(args["output_dir"] +"/"+ args["method"])
        
    if os.path.exists(args["data_dir"]):
        images_list = sorted(os.listdir(args["data_dir"]))
        # print(images_list)
        image_sets = [name.split("_")[0] for name in images_list]
        unique_set = np.array(image_sets)
        unique_set = list(np.unique(unique_set))        
        for pair in unique_set:
            image1 = cv.imread(args["data_dir"] +"/"+ pair + "_10.png",1)
            image2 = cv.imread(args["data_dir"] +"/"+ pair + "_11.png",1)
            # print(image1.shape,image2.shape)
            flow, mask = optical_flow(image1,image2, method = args["method"])
            save_data(args["output_dir"] +"/"+ args["method"]+"/"+ pair + "_10.png", flow, mask)

        if args["eval"]:
            evalaute_data(args["output_dir"],method = args["method"])
        print("output dir:",args["output_dir"] +"/"+ args["method"])
        print("done")
    else: 
        print("data directory given as input is empty (invalid data)")
        return 

if __name__ == "__main__":
    Methods_list = ["lucas_kanade_Fast_features",
                    "lucas_kanade_GoodFeaturesToTrack",
                    "Farneback_flow",
                    "spynet"]
    print("list of methods")
    for methods in Methods_list:
        print(methods)
        
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data_dir", required=False, default="./dataset/data_scene_flow/training/image_2",
        help="path to input dataset (i.e., directory of images)")
    
    ap.add_argument("-o", "--output_dir", required=False, default= "./results",
        help="path of directory for the results")

    ap.add_argument("-m", "--method", required=False, default= "spynet",
        help="method")
    
    ap.add_argument("--eval",type = bool,default=False)
    
    args = vars(ap.parse_args())
    
    calculate_flow(args)