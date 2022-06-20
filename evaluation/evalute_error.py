import numpy as np
import os
import pandas as pd
import cv2 as cv

def evalute_kitti_error(path):
    list_of_files = sorted(os.listdir(path + "/errors_flow_noc"))
    # print(list_of_files)
    total_data = []
    for i in list_of_files:
        if(i[-4:] == ".txt"):
            current_data = np.loadtxt(path + "/errors_flow_noc" + "/" + i)  
            total_data.append(current_data)
    total_data = np.array(total_data).reshape(-1,7)
    data = {"errors_bg" : total_data[:,1],
            "errors_fg" : total_data[:,3],
            "errors_all": total_data[:,5]
            }
    df = pd.DataFrame.from_dict(data)
    df.to_csv(path + "/errors_flow_noc" + "/error_data.csv")
    print(df.describe())
    # print("evaluting AAE,EPE,absolute error")
    list_of_flow_maps = sorted(os.listdir(path))
    gt_path = "dataset/data_scene_flow/training/flow_noc"
    total_AAE = []
    total_EPE = []
    total_absolute_error = []
    for i in list_of_flow_maps:
        if(i[-4:] == ".png"):
            gt_flow = cv.imread(gt_path + "/" + i, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
            # print(gt_flow)
            result_flow = cv.imread(path + "/" + i, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
            error_map, AAE, EPE, absolute_error = error_kitti(gt_flow,result_flow)
            if not os.path.exists(path + "/error_vis"):
                os.makedirs(path + "/error_vis")
            cv.imwrite(path + "/error_vis/" + i ,error_map)
            total_AAE.append(AAE)
            total_EPE.append(EPE)
            total_absolute_error.append(absolute_error)
    data_error = {
        "AAE" : total_AAE,
        "EPE" : total_EPE,
        "absolute_error": total_absolute_error
    }
    df = pd.DataFrame.from_dict(data_error)
    df.to_csv(path + "/errors_flow_noc" + "/error_data2.csv")
    print(df.describe())
    
def error_kitti(gt_flow,result_flow):
    """calculate error maps and AAE,EPE,absolute_error

    Args:
        gt_flow (_type_): _description_
        result_flow (_type_): _description_
    """
    mask = result_flow[:,:,0]*gt_flow[:,:,0]
    count = np.count_nonzero(mask)
    actual_gt = gt_flow[:,:,1:]/64
    actual_gt = actual_gt - 512
    actual_result = result_flow[:,:,1:]/64
    actual_result = actual_result - 512
    error_map = (actual_gt - actual_result)
    error_map[:,:,0] = error_map[:,:,0]*mask
    error_map[:,:,1] = error_map[:,:,1]*mask
    num = actual_gt[:,:,0]*actual_result[:,:,0] + actual_gt[:,:,1]*actual_result[:,:,1] + 1
    num = num*mask
    den = ((np.square(actual_gt[:,:,0])+ np.square(actual_gt[:,:,1])+ 1)*(np.square(actual_result[:,:,0])+ np.square(actual_result[:,:,1])+ 1))*mask
    den = np.sqrt(den)
    AAE = np.sum(np.arccos(num/(den + 0.000000001)))*(1/(count+0.0000000001))
    EPE = np.sum(np.sqrt(np.square(error_map[:,:,0])+ np.square(error_map[:,:,1])))*(1/count)
    absolute_error = np.sum(np.abs(error_map[:,:,0]) + np.abs(error_map[:,:,1]))*(1/count)
    hsv = np.zeros((error_map.shape[0], error_map.shape[1], 3), dtype=np.uint8)
    error_map_magnitude, error_map_angle = cv.cartToPolar(error_map[..., 0].astype(np.float32), error_map[..., 1].astype(np.float32))

    nans = np.isnan(error_map_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        error_map_magnitude[nans] = 0.
        
    hsv[..., 0] = error_map_angle * 180 / np.pi / 2
    hsv[..., 1] = cv.normalize(error_map_magnitude, None, 0, 255, cv.NORM_MINMAX)
    hsv[..., 2] = 255
    error_map = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    return error_map, AAE, EPE, absolute_error

def error_vkitti(gt_flow,result_flow):
    w,h,_ = gt_flow.shape
    mask = gt_flow[:,:,0]*(1/(2**16 - 1))
    count = np.count_nonzero(mask) 
    
    actual_gt = gt_flow[:,:,1:]*(1/(2**16 - 1))
    actual_gt = actual_gt*2 - 1
    actual_gt[:,:,0] = actual_gt[:,:,0]*(h-1)
    actual_gt[:,:,1] = actual_gt[:,:,1]*(w-1)
    
    actual_result = result_flow[:,:,1:]*(1/(2**16 - 1))
    actual_result = actual_result*2 - 1
    actual_result[:,:,0] = actual_result[:,:,0]*(h-1)
    actual_result[:,:,1] = actual_result[:,:,1]*(w-1)
    
    error_map = (actual_gt - actual_result)
    error_map[:,:,0] = error_map[:,:,0]*mask
    error_map[:,:,1] = error_map[:,:,1]*mask
    num = actual_gt[:,:,0]*actual_result[:,:,0] + actual_gt[:,:,1]*actual_result[:,:,1] + 1
    num = num*mask
    den = ((np.square(actual_gt[:,:,0])+ np.square(actual_gt[:,:,1])+ 1)*(np.square(actual_result[:,:,0])+ np.square(actual_result[:,:,1])+ 1))*mask
    den = np.sqrt(den)
    AAE = np.sum(np.arccos(num/(den + 0.000000001)))*(1/(count+0.0000000001))
    EPE = np.sum(np.sqrt(np.square(error_map[:,:,0])+ np.square(error_map[:,:,1])))*(1/count)
    absolute_error = np.sum(np.abs(error_map[:,:,0]) + np.abs(error_map[:,:,1]))*(1/count)
    hsv = np.zeros((error_map.shape[0], error_map.shape[1], 3), dtype=np.uint8)
    error_map_magnitude, error_map_angle = cv.cartToPolar(error_map[..., 0].astype(np.float32), error_map[..., 1].astype(np.float32))

    nans = np.isnan(error_map_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        error_map_magnitude[nans] = 0.
        
    hsv[..., 0] = error_map_angle * 180 / np.pi / 2
    hsv[..., 1] = cv.normalize(error_map_magnitude, None, 0, 255, cv.NORM_MINMAX)
    hsv[..., 2] = 255
    error_map = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)
    return error_map, AAE, EPE, absolute_error    

def evaluate_vkitti_error(seq):
    path = "/home/sushlok/optical_flow/GMFlowNet/result/0020/vkitti_{}/".format(seq)
    list_of_flow_maps = sorted(os.listdir(path))
    gt_path = "/home/sushlok/new_approach/datasets/vkitti/vkitti_1.3.1_flowgt/0020/{}".format(seq)
    total_AAE = []
    total_EPE = []
    total_absolute_error = []
    for i in list_of_flow_maps:
        if(i[-4:] == ".png"):
            gt_flow = cv.imread(gt_path + "/" + i, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
            # print(gt_flow)
            result_flow = cv.imread(path + "/" + i, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
            error_map, AAE, EPE, absolute_error = error_kitti(gt_flow,result_flow)
            if not os.path.exists(path + "/error_vis"):
                os.makedirs(path + "/error_vis")
            cv.imwrite(path + "/error_vis/" + i ,error_map)
            total_AAE.append(AAE)
            total_EPE.append(EPE)
            total_absolute_error.append(absolute_error)
    data_error = {
        "AAE" : total_AAE,
        "EPE" : total_EPE,
        "absolute_error": total_absolute_error
    }
    df = pd.DataFrame.from_dict(data_error)
    df.to_csv(path + "/error_data2.csv")
    print(df.describe())


if __name__ == "__main__":
    Methods_list = [
        "15-deg-left",
        "15-deg-right",
        "30-deg-left",
        "30-deg-right",
        "clone",
        "fog",
        "morning",
        "overcast",
        "rain",
        "sunset"
    ]
    
    for method in Methods_list:
        print(method)
        evaluate_vkitti_error(method)
