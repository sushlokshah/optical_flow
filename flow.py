import numpy as np
import cv2 as cv 
import os 
import sys
import argparse
import matplotlib.pyplot as plt
import time
import torch
import pandas as pd
# import numba

from classical_flow.classical_flow_methods import lucus_kanade_flow, Farneback_flow
from spynet.run import estimate
from evaluation.evalute_error import evalute_kitti_error

class execution_data():
    def __init__(self):
        self.images = []
        self.features = []
        self.feature_extraction_time = []
        self.flow_estimation_time = []
        self.total_time = []
        # dynamic_cpu_memory_consumption = []
        # dynamic_gpu_memory_consumption = []
    
    def save_stats(self, csv_file_path):
        num_features = []
        for i in range(len(self.features)):
            if np.array(self.features[i]).all() != None:
                num_features.append(len(self.features[i]))
        data = {"images" : self.images,
        "features" : num_features,
        "feature_extraction_time" : self.feature_extraction_time,
        "flow_estimation_time" : self.flow_estimation_time,
        "total_time" : self.total_time}
        df = pd.DataFrame.from_dict(data)
        df.to_csv(csv_file_path)
        print("time:\n", df.describe())
        # df = pd.read_csv(csv_file_path)
                
def data_formating(flow, mask):
    flow = (flow + 512)*64
    flow_kitti_format = np.zeros([flow.shape[0],flow.shape[1],3])
    flow_kitti_format[:,:,2] = flow[:,:,0]
    flow_kitti_format[:,:,1] = flow[:,:,1]
    flow_kitti_format[:,:,0] = mask
    # plt.imshow(flow_kitti_format)
    # plt.show()
    return flow_kitti_format

def save_data(path,mask_path,flow,mask):
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
    mask = np.array(mask).astype(np.int64)
    cv.imwrite(mask_path,mask)
    flow_kitti_format = flow_kitti_format.astype(np.uint16)
    cv.imwrite(path,flow_kitti_format)

def optical_flow(image1,image2,params, method = None):
    if method == "lucas_kanade_GoodFeaturesToTrack":
        lkgft = params["lucas_kanade_GoodFeaturesToTrack_params"]
        lk_params = dict( winSize  = lkgft["winSize"][0],
                    maxLevel = lkgft["maxLevel"][0],
                    criteria = lkgft["criteria"])
        feature_params = dict( maxCorners = lkgft["maxCorners"][0],
                            qualityLevel = lkgft["qualityLevel"][0],
                            minDistance = lkgft["minDistance"][0],
                            blockSize = lkgft["blockSize"] )
        flow, mask, features,feature_extraction_time_, flow_estimation_time_= lucus_kanade_flow(image1,image2,lk_params,feature_params,method="GoodFeaturesToTrack")
    elif method == "lucas_kanade_Fast_features":
        lkfft = params["lucas_kanade_Fast_features_params"]
        lk_params = dict( winSize  = lkfft["winSize"][0],
                    maxLevel = lkfft["maxLevel"][0],
                    criteria = lkfft["criteria"])
        # print(lk_params)
        feature_params = {}
        feature_params["NonmaxSuppression"] = lkfft["NonmaxSuppression"]
        feature_params["Threshold"] = lkfft["Threshold"]
        flow, mask, features,feature_extraction_time_, flow_estimation_time_ = lucus_kanade_flow(image1,image2,lk_params,feature_params,method="Fast_features")
    elif method == "Farneback_flow":
        feature_extraction_time_ = 0
        features = []
        flow, mask,flow_estimation_time_ = Farneback_flow(image1,image2,params["Farneback_flow_params"])
    elif method == "spynet":
        features = []
        feature_extraction_time_ = 0
        # print(torch.FloatTensor((image1/255).transpose(2, 0, 1)).shape)
        start = time.time()
        flow = estimate(torch.FloatTensor((image1/255).transpose(2, 0, 1)),torch.FloatTensor((image1/255).transpose(2, 0, 1)))
        end = time.time()
        flow_estimation_time_ = end - start
        flow = flow.numpy().transpose(1, 2, 0)
        mask = np.ones([flow.shape[0],flow.shape[1]])
    return flow, mask, features,feature_extraction_time_, flow_estimation_time_

def evalaute_data(path, method = None):
        # os.system("g++ -O3 -DNDEBUG -o ./evalution/cpp/evaluate_scene_flow ./evalution/cpp/evaluate_scene_flow.cpp -lpng")
        os.system("./evaluation/cpp/build/evaluate_scene_flow "+ method)
        print(path + "/" + method)
        evalute_kitti_error(path + "/" + method)

def calculate_flow(args,params):
    if not os.path.exists(args["output_dir"] +"/"+ args["method"]):
        os.makedirs(args["output_dir"] +"/"+ args["method"])
    
    if not os.path.exists(args["output_dir"] +"/"+ args["method"] + "/" + "masks"):
        os.makedirs((args["output_dir"] +"/"+ args["method"]+ "/" + "masks"))
        
    if os.path.exists(args["data_dir"]):
        images_list = sorted(os.listdir(args["data_dir"]))
        # print(images_list)
        # total_exec_time_list = []
        # feaatures = []
        # feature_extraction_time = []
        data = execution_data()

        image_sets = [name.split("_")[0] for name in images_list]
        unique_set = np.array(image_sets)
        unique_set = list(np.unique(unique_set))        
        for pair in unique_set:
            image1 = cv.imread(args["data_dir"] +"/"+ pair + "_10.png",1)
            image2 = cv.imread(args["data_dir"] +"/"+ pair + "_11.png",1)
            # print(image1.shape,image2.shape)
            flow, mask, features,feature_extraction_time_, flow_estimation_time_ = optical_flow(image1,image2,params, method = args["method"])
            data.images.append(args["data_dir"] +"/"+ pair + "_10.png")
            data.features.append(features)
            data.feature_extraction_time.append(feature_extraction_time_)
            data.flow_estimation_time.append(flow_estimation_time_)
            data.total_time.append(feature_extraction_time_ + flow_estimation_time_)
            
            # total_exec_time_list.append(end -start)
            save_data(args["output_dir"] +"/"+ args["method"]+"/"+ pair + "_10.png",args["output_dir"] +"/"+ args["method"]+"/" + "masks/"+ pair + "_10.png", flow, mask)

        # np.savez(args["output_dir"] +"/"+ args["method"] + ".npz" ,np.array(exec_time_list) )
        data.save_stats(args["output_dir"] +"/"+ args["method"]+"/time_stats.csv")
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
    
    """set these parameters for the setting execution configuration.
    """
    Farneback_flow_params = {}
    Farneback_flow_params["pyr_scale"] =  0.5
    Farneback_flow_params["levels"] = 3
    Farneback_flow_params["winsize"] = 15
    Farneback_flow_params["iterations"] = 3
    Farneback_flow_params["poly_n"] = 5
    Farneback_flow_params["poly_sigma"] = 1.2 
    Farneback_flow_params["flags"] = 0

    lucas_kanade_Fast_features_params = {}
    lucas_kanade_Fast_features_params["NonmaxSuppression"] = 1
    lucas_kanade_Fast_features_params["Threshold"] = 30
    lucas_kanade_Fast_features_params["winSize"] = (15, 15),
    lucas_kanade_Fast_features_params["maxLevel"] = 2,
    lucas_kanade_Fast_features_params["criteria"] = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
    
    lucas_kanade_GoodFeaturesToTrack_params = {}
    lucas_kanade_GoodFeaturesToTrack_params["maxCorners"] = 10000,
    lucas_kanade_GoodFeaturesToTrack_params["qualityLevel"] = 0.1,
    lucas_kanade_GoodFeaturesToTrack_params["minDistance"] = 3,
    lucas_kanade_GoodFeaturesToTrack_params["blockSize"] = 3
    lucas_kanade_GoodFeaturesToTrack_params["winSize"] = (15, 15),
    lucas_kanade_GoodFeaturesToTrack_params["maxLevel"] = 2,
    lucas_kanade_GoodFeaturesToTrack_params["criteria"] = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
    
    params = {}
    params["Farneback_flow_params"] = Farneback_flow_params
    params["lucas_kanade_Fast_features_params"] = lucas_kanade_Fast_features_params
    params["lucas_kanade_GoodFeaturesToTrack_params"] = lucas_kanade_GoodFeaturesToTrack_params
    
    ap = argparse.ArgumentParser()
    ap.add_argument("-d","--data_dir", required=False, default="./dataset/data_scene_flow/training/image_2",
        help="path to input dataset (i.e., directory of images)")
    
    ap.add_argument("-o", "--output_dir", required=False, default= "./results",
        help="path of directory for the results")

    ap.add_argument("-m", "--method", required=False, default= "spynet",
        help="method")
    
    ap.add_argument("--eval",type = bool,default=True)
    
    args = vars(ap.parse_args())
    
    calculate_flow(args, params)
    
    
# count   200.000000               200.000000            200.000000  200.000000
# mean   1091.860000                 0.005558              0.002215    0.007773
# std     658.928558                 0.001294              0.001130    0.001953
# min      61.000000                 0.004895              0.000913    0.005960
# 25%     639.000000                 0.005197              0.001458    0.006727
# 50%     944.000000                 0.005363              0.001823    0.007197
# 75%    1319.750000                 0.005552              0.002621    0.008182
# max    4507.000000                 0.021938              0.006987    0.027318 




# time:
#            features  feature_extraction_time  flow_estimation_time  total_time
# count   200.000000               200.000000            200.000000  200.000000
# mean   3142.865000                 0.002380              0.004992    0.007372
# std    1586.492309                 0.003108              0.002426    0.004687
# min      84.000000                 0.000226              0.000760    0.000985
# 25%    1986.250000                 0.001490              0.003134    0.004565
# 50%    2770.500000                 0.001970              0.004476    0.006396
# 75%    3845.250000                 0.002759              0.006022    0.008901
# max    9374.000000                 0.044020              0.014855    0.055156

# ./results/lucas_kanade_Fast_features
#        errors_bg  errors_fg  errors_all
# count   2.000000   2.000000    2.000000
# mean    0.374854   0.408638    0.384918
# std     0.039570   0.006973    0.029664
# min     0.346874   0.403707    0.363943
# 25%     0.360864   0.406172    0.374431
# 50%     0.374854   0.408638    0.384918
# 75%     0.388845   0.411104    0.395406
# max     0.402835   0.413569    0.405894

# ./results/lucas_kanade_Fast_features
# evaluting AAE,EPE,absolute error
#                 AAE         EPE  absolute_error
# count    200.000000  200.000000      200.000000
# mean    2299.834379   16.176276       19.418795
# std     1853.132945   13.490975       16.248685
# min      544.028565    0.332128        0.403185
# 25%     1469.188801    4.911900        5.834550
# 50%     2003.000418   13.437732       16.423746
# 75%     2579.586602   24.651536       29.693925
# max    22861.001423   74.758801       86.021432


#                AAE         EPE  absolute_error
# count   200.000000  200.000000      200.000000
# mean    576.900102   18.038368       21.989338
# std     636.860607   13.212509       16.147871
# min     191.537988    0.300616        0.368305
# 25%     402.608269    5.721263        6.968899
# 50%     499.161542   16.528652       20.312301
# 75%     637.373469   28.659821       35.502201
# max    9143.668797   63.599354       72.990135