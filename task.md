The goal of the internship is to research and quantitatively evaluate different optical flow methods. The dataset of choice is the Kitti Optical Flow 2015 dataset (http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow). The following assignments are a guideline on how to proceed. This guideline is subject to change and can be adapted throughout the internship.

- Download the Kitti dataset and get accustomed to it. Explore the optical flow devkit and see if it is useful. Skim the paper (http://www.cvlibs.net/publications/Menze2015CVPR.pdf).

- Implement a python script that performs optical flow on the Kitti dataset scenes. As a primary image programming framework use OpenCV-Python. The script should cycle through the scenes and perform optical flow calulation on a sparse set of image features. The optical flow component should be exchangable as well as the feature detector. 

- a) As features use Fast-features and GoodFeaturesToTrack of OpenCV. 
- b) As an optical flow method baseline the Lukas-Kanade Tracker (https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_py_lucas_kanade.html).

- Once the pipeline is running machine learning optical flow methods should be researched that will be compared to the baseline approach. For this you might need to install the CUDA toolkit. Determine which CUDA toolkit version is compatible with the given graphics card. When the code for a specific method is given verify that the weights of the network is also available. If retraining is necessary don't investigate the approach further. 

A few useful links for optical flow methods:

- http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow
Check out the methods, when the code is given. 

- https://github.com/open-mmlab/mmflow
Compiles a few deep learning optical flow methods.

- https://paperswithcode.com/task/optical-flow-estimation
Click through some high performing optical flow methods and see which ones are interesting. 

Contact me through discord during all steps whenever help is needed. 