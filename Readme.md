# Optical Flow

## setup:
1. Download Kitti Optical Flow Dataset
2. git clone https://github.com/sushlokshah/optical_flow.git
3. cd optical_flow/
4. mkdir dataset
5. extract the dataset in this folder
    ```
    directory structure
        |-- dataset
            |-- data_scene_flow
                |-- testing
                |-- training
    ```
6. mkdir results
7. go the root directory and download dependencies
```
conda env create -f environment.yml
```
8. create executables of cpp code
```
cd evaluation/cpp
mkdir build
cd build
cmake ..
make
```
9. For estimating the results: execute the python file with the appropriate method.
for example.
```
python flow.py -m Farneback_flow
```

* Hackmd link: https://hackmd.io/@Sushlok/S1FMH1Wv5/edit

* kitti result format
 - num_errors_bg/num_pixels_bg
 - num_errors_bg_result/num_pixels_bg_result
 - num_errors_fg/num_pixels_fg
 - num_errors_fg_result/num_pixels_fg_result
 - num_errors_all/num_pixels_all
 - num_errors_all_result/num_pixels_all_result

* Results: [Sheet_link](https://docs.google.com/spreadsheets/d/15hT6XEFKs1q0NTeMnrbii8cBUI5k4g700TiIjzfawUo/edit?usp=sharing) 
* reviewed papers: [drive link](https://drive.google.com/drive/folders/17CLwbwz5EvTkkyQUpKUS2e3JTBqqEXK6?usp=sharing)