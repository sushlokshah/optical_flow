from mmflow.apis import inference_model, init_model
import cv2 as cv
import numpy as np
config_file = 'mmflow/configs/pwcnet/pwcnet_ft_4x1_300k_sintel_final_384x768.py'
checkpoint_file = 'mmflow/mim/pwcnet_ft_4x1_300k_sintel_final_384x768.pth'
device = 'cuda:0'
# init a model
model = init_model(config_file, checkpoint_file, device=device)
# inference the demo image
image = inference_model(model, 'mmflow/demo/frame_0001.png', 'mmflow/demo/frame_0002.png')
flow = np.zeros([image.shape[0],image.shape[1],3])
flow[:,:,0] = 1
flow[:,:,1:] = image
cv.imwrite("flow_out.png",flow)