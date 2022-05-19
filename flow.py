import numpy as np
import cv2 as cv 
import os 
import sys
import argparse
import matplotlib.pyplot as plt

from classical_flow.classical_flow_methods import lucus_kanade_flow, Farneback_flow

def optical_flow(image1,image2, method = None):
    if method == "lucas_kanade_GoodFeaturesToTrack":
        flow = lucus_kanade_flow(image1,image2,method="GoodFeaturesToTrack")
    elif method == "lucas_kanade_Fast_features":
        flow = lucus_kanade_flow(image1,image2,method="Fast_features")
    elif method == "Farneback_flow":
        flow = Farneback_flow(image1,image2)
    return flow

def save_data(path,flow):
    # flow = flow.astype(np.int16)
    # print(type(flow))
    cv.imwrite(path,flow)

def evalaute_data(path, method = None):
    if method == "lucas_kanade_Fast_features" or method == "lucas_kanade_GoodFeaturesToTrack":
        print("evalution script is not ready yet")
    
    else:
        print("running kitti benchmark")
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
            print(image1.shape,image2.shape)
            flow = optical_flow(image1,image2, method = args["method"])
            save_data(args["output_dir"] +"/"+ args["method"]+"/"+ pair + "_10.png", flow)

        if args["eval"]:
            evalaute_data(args["output_dir"],method = args["method"])
            
    else: 
        print("data directory given as input is empty (invalid data)")
        return 

if __name__ == "__main__":
    Methods_list = ["lucas_kanade_Fast_features",
                    "lucas_kanade_GoodFeaturesToTrack",
                    "Farneback_flow"]
    print("list of methods")
    for methods in Methods_list:
        print(methods)
        
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--data_dir", required=False, default="./dataset/data_scene_flow/training/image_2",
        help="path to input dataset (i.e., directory of images)")
    
    ap.add_argument("-o", "--output_dir", required=False, default= "./results",
        help="path of directory for the results")

    ap.add_argument("-m", "--method", required=False, default= "Farneback_flow",
        help="method")
    
    ap.add_argument("--eval",type = bool,default=True)
    
    args = vars(ap.parse_args())
    
    calculate_flow(args)