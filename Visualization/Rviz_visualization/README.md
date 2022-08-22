# Rviz_visualization


## Introduction

Rviz_visualization is a project which use ROS to publish Pointcloud/Image/3Dbox/Speed data in rviz. It is convenient to visualize the result in rviz.

## Major features
* Support many data type. 
```
    Pointcloud
    Image
    3Dbox
    Speed
```
* Support different file suffix.
```
    Pointcloud : pcd, pkl, npy, bin, txt
    Image : png, jpg...
    3Dbox : txt, pkl
```

### Attention
Pkl file is only for waymo-det3d format, reference [det3d-waymo](https://github.com/tianweiy/CenterPoint/blob/master/docs/WAYMO.md),
    and only works for python3!

## Installation

requirements
```
    python2.7/python3.6+
    numpy
    opencv-python
    rospy
    pickle(only python3.6)
```

## Usage

1、Prepare your own data.
```
└── XXX 
    ├── lidar <-- all lidar files(pcd/npy/bin/txt/pkl...) 
    ├── image <-- all image files(jpg/png/bmp...) 
    ├── annos <-- all label files(pkl/txt...)
    ├── poses <-- all pose files(pkl/txt...)
```
2、Run in terminal.
```
    python main.py \
            --lidar_path /mnt/data/SGData/20220120/2022-01-20-09-52-09-morning_inv/lidar_npy \
            --image_path /mnt/data/SGData/20220120/2022-01-20-09-52-09-morning_inv/cam60 \
            --rate 50 \
            --fov 60
```
3、Open rviz
```
    Open rviz in terminal, and add topics as below:
        /my_point_cloud
        /my_image
        /my_box3d
        /my_ego_view
```
4、Demo

![gif](rviz_sgdata.gif)

* Attention: main.py can only visualize lidar & image, and the sort function is default. If your files order is not by str, please modify the sort function!!!

