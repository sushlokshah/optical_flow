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
    # print(df.describe())
    print("evaluting AAE,EPE,absolute error")
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
            error_map, AAE, EPE, absolute_error = error(gt_flow,result_flow)
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
    
def error(gt_flow,result_flow):
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

if __name__ == "__main__":
    Methods_list = [
        "Farneback_flow",
    "Farneback_flow_Fast_features",
    "lucas_kanade_GoodFeaturesToTrack",
    "Farneback_flow_GoodFeaturesToTrack"  ,
    "spynet",
    "flownet2",
    "spynet_Fast_features",
    "flownet2_Fast_features",
    "spynet_GoodFeaturesToTrack",
    "flownet2_GoodFeaturesToTrack",
    "lucas_kanade_Fast_features"
    ]
    
    for method in Methods_list:
        print(method)
        evalute_kitti_error("results/"+ method)

# Farneback_flow
# evaluting AAE,EPE,absolute error
#               AAE         EPE  absolute_error
# count  200.000000  200.000000      200.000000
# mean     8.568464   18.413545       21.993941
# std      2.505330   13.049043       15.550219
# min      5.217399    0.576749        0.689025
# 25%      6.743976    6.775437        7.433892
# 50%      7.777092   16.770607       19.891850
# 75%     10.162346   27.554160       33.158613
# max     18.207155   65.316676       73.785977
# Farneback_flow_Fast_features
# evaluting AAE,EPE,absolute error
#                 AAE         EPE  absolute_error
# count    200.000000  200.000000      200.000000
# mean    7096.294802   13.997604       16.216390
# std     5116.747307   14.289669       16.738993
# min     1827.879143    0.297945        0.357838
# 25%     3959.638923    2.691622        3.044450
# 50%     5736.707427    8.186814        9.894705
# 75%     8312.316618   21.007835       24.024566
# max    36578.535683   62.650080       71.110156
# lucas_kanade_GoodFeaturesToTrack
# evaluting AAE,EPE,absolute error
#                 AAE         EPE  absolute_error
# count    200.000000  200.000000      200.000000
# mean    7653.564592   15.276718       18.180049
# std     4889.991477   15.031831       17.960933
# min     1846.635664    0.278265        0.345219
# 25%     4258.805699    3.732687        4.508292
# 50%     6388.306123   11.165322       13.224651
# 75%     9143.686557   22.228016       25.722621
# max    33253.056388   79.954894       91.178480
# Farneback_flow_GoodFeaturesToTrack
# evaluting AAE,EPE,absolute error
#                 AAE         EPE  absolute_error
# count    200.000000  200.000000      200.000000
# mean    7653.455244   14.030820       16.237749
# std     4889.990467   14.774007       17.374391
# min     1846.281700    0.308523        0.378873
# 25%     4258.579592    2.664303        3.104672
# 50%     6388.469237    8.558457        9.801835
# 75%     9143.575317   21.398436       24.184525
# max    33253.064943   66.049756       80.201778
# spynet
# evaluting AAE,EPE,absolute error
#               AAE         EPE  absolute_error
# count  200.000000  200.000000      200.000000
# mean     9.491162   27.780398       33.560863
# std      2.454118   14.498858       17.617940
# min      5.474261    1.374630        1.571077
# 25%      7.814526   15.615274       17.917794
# 50%      8.708261   28.260279       34.304103
# 75%     11.031913   39.503702       48.650509
# max     18.943077   66.825574       76.509798
# flownet2
# evaluting AAE,EPE,absolute error
#               AAE         EPE  absolute_error
# count  200.000000  200.000000      200.000000
# mean     8.263890    7.216495        9.005439
# std      2.485370    5.929928        7.283033
# min      4.902330    0.375483        0.450577
# 25%      6.455091    2.634155        3.244717
# 50%      7.427283    5.949977        7.214701
# 75%      9.901683   10.170264       12.668404
# max     17.525063   38.781542       45.951354
# spynet_Fast_features
# evaluting AAE,EPE,absolute error
#                 AAE         EPE  absolute_error
# count    200.000000  200.000000      200.000000
# mean    7097.314793   23.810993       27.688995
# std     5116.740279   15.417653       18.098461
# min     1828.923374    1.963656        2.400145
# 25%     3959.980496   11.153252       12.630591
# 50%     5737.785464   20.319770       24.157753
# 75%     8313.487896   32.754944       38.123433
# max    36579.750096   77.555418       83.955560
# flownet2_Fast_features
# evaluting AAE,EPE,absolute error
#                 AAE         EPE  absolute_error
# count    200.000000  200.000000      200.000000
# mean    7096.145880    5.461960        6.455801
# std     5116.764729    4.530133        5.265460
# min     1827.465132    0.507351        0.577945
# 25%     3958.899691    1.888250        2.261562
# 50%     5736.700554    4.261946        5.064280
# 75%     8312.123855    8.620844        9.974417
# max    36578.483877   27.374010       29.731519
# spynet_GoodFeaturesToTrack
# evaluting AAE,EPE,absolute error
#                 AAE         EPE  absolute_error
# count    200.000000  200.000000      200.000000
# mean    7654.472499   23.697483       27.501816
# std     4889.982999   15.960535       18.767102
# min     1847.407964    2.051896        2.498779
# 25%     4259.336650   11.007867       12.536731
# 50%     6389.114068   19.463704       22.698479
# 75%     9144.787558   32.071075       37.252266
# max    33254.171527   79.127151       90.185884
# flownet2_GoodFeaturesToTrack
# evaluting AAE,EPE,absolute error
#                 AAE         EPE  absolute_error
# count    200.000000  200.000000      200.000000
# mean    7653.308858    5.423069        6.387727
# std     4890.020126    4.675617        5.467402
# min     1845.939364    0.429206        0.522059
# 25%     4258.062080    2.036647        2.376017
# 50%     6388.347197    4.038234        4.767426
# 75%     9143.578119    8.083281        9.341403
# max    33253.168197   30.865951       36.145522
# lucas_kanade_Fast_features
# evaluting AAE,EPE,absolute error
#                 AAE         EPE  absolute_error
# count    200.000000  200.000000      200.000000
# mean    7096.394593   15.051478       17.955422
# std     5116.741622   14.274599       17.092380
# min     1828.189378    0.275271        0.345052
# 25%     3959.305330    3.786542        4.373361
# 50%     5736.761779   11.468896       12.964385
# 75%     8312.313866   22.132487       26.926398
# max    36578.681768   77.021834       87.980827


# flownet2_Fast_features
# evaluting AAE,EPE,absolute error
#                 AAE         EPE  absolute_error
# count    200.000000  200.000000      200.000000
# mean    2299.536530    5.743285        6.908074
# std     1853.168276    4.553046        5.441535
# min      543.212603    0.523004        0.655417
# 25%     1468.864690    2.150432        2.546601
# 50%     2002.919051    4.593777        5.440471
# 75%     2579.354111    8.399910       10.052822
# max    22861.102668   29.275802       33.127447


# spynet_Fast_features
# evaluting AAE,EPE,absolute error
#                 AAE         EPE  absolute_error
# count    200.000000  200.000000      200.000000
# mean    2300.724367   24.521518       28.853539
# std     1853.133563   14.703993       17.540454
# min      544.731371    1.948601        2.361130
# 25%     1470.293843   11.978296       13.537248
# 50%     2003.901460   23.025346       26.480601
# 75%     2580.503800   34.157752       40.675643
# max    22862.163303   74.111003       79.245417


# Farneback_flow_Fast_features
# evaluting AAE,EPE,absolute error
#                 AAE         EPE  absolute_error
# count    200.000000  200.000000      200.000000
# mean    2299.709467   14.428517       16.886804
# std     1853.147710   13.195477       15.651136
# min      543.763080    0.348487        0.419212
# 25%     1469.046001    3.219420        3.598748
# 50%     2002.980187   10.883405       12.532676
# 75%     2579.588907   23.703248       26.711227
# max    22861.020262   59.214656       63.999748













# Farneback_flow_Fast_features
# evaluting AAE,EPE,absolute error
#                AAE         EPE  absolute_error
# count   200.000000  200.000000      200.000000
# mean    576.728604   14.609299       17.505646
# std     636.881624   11.773839       14.274983
# min     191.139590    0.366462        0.444160
# 25%     402.457557    3.391124        4.038764
# 50%     499.068802   12.458332       14.673617
# 75%     637.240623   24.042538       28.455438
# max    9143.687351   49.805150       53.818605


# spynet_Fast_features
# evaluting AAE,EPE,absolute error
#                AAE         EPE  absolute_error
# count   200.000000  200.000000      200.000000
# mean    577.731774   25.138208       30.446071
# std     636.870668   14.151296       17.336155
# min     192.482144    1.640613        1.927235
# 25%     403.520432   12.533387       14.709884
# 50%     500.165270   25.222972       31.067501
# 75%     638.074906   37.225231       45.859840
# max    9144.728728   65.013639       68.538857
# flownet2_Fast_features
# evaluting AAE,EPE,absolute error
#                AAE         EPE  absolute_error
# count   200.000000  200.000000      200.000000
# mean    576.533763    6.467952        8.072568
# std     636.906794    4.904442        6.199342
# min     190.993019    0.372727        0.439498
# 25%     402.073152    2.424859        2.953812
# 50%     498.764230    5.257110        6.512272
# 75%     636.995870    9.727979       12.289408
# max    9143.804499   30.967580       38.093980