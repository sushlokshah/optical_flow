import numpy as np
import cv2 as cv 
import os

dense_methods = ["Farneback_flow",
                    "spynet",
                    "flownet2"]

sparse_methods = ["lucas_kanade_Fast_features",
                    "lucas_kanade_GoodFeaturesToTrack"
                    ]

result_dir = "results"
    
list_img = sorted(os.listdir(result_dir + "/"+ "lucas_kanade_Fast_features"))
for path in list_img:
    if(path[-4:] == ".png"):
        print(path)
        for method_s in sparse_methods:
            for method_d in dense_methods:
                sparse_flow = cv.imread(result_dir+ "/"+ method_s + "/" + path, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
                dense_flow = cv.imread(result_dir+ "/"+ method_d + "/" + path, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
                # print(sparse_flow.shape)
                if(sparse_flow.shape != dense_flow.shape):
                    print(result_dir +"/"+ method_d + "/"+ path)
                    dense_flow = cv.resize(dense_flow, (sparse_flow.shape[1],sparse_flow.shape[0]), interpolation = cv.INTER_AREA)
                    cv.imwrite(result_dir +"/"+ method_d + "/"+ path,dense_flow)
                dense_flow[:,:,0] = sparse_flow[:,:,0]
                if(method_s == "lucas_kanade_Fast_features"):
                    if not os.path.exists(result_dir +"/"+ method_d + "_Fast_features"):
                        os.makedirs(result_dir +"/"+ method_d + "_Fast_features")
                    cv.imwrite(result_dir +"/"+ method_d + "_Fast_features/"+ path,dense_flow)
                else:
                    if not os.path.exists(result_dir +"/"+ method_d + "_GoodFeaturesToTrack"):
                        os.makedirs(result_dir +"/"+ method_d + "_GoodFeaturesToTrack")
                    cv.imwrite(result_dir +"/"+ method_d + "_GoodFeaturesToTrack/"+ path,dense_flow)