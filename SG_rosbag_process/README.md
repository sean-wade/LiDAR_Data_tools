# SG_rosbag_process

## Instruction
* 本代码库为处理 SG-rosbag data 的功能库
* 其中最主要的是 rosbag_extractor_align_image_undistort.py 代码，其作用是从 rosbag 中抽取关键帧（去畸变、对齐）后送标使用

## requirements
* python-ros
* python-pcl
* opencv-python

## Update
[2022-06-17] 更新了 rosbag_extractor_align_image_undistort.py, 主要更改为:
```
        1. 图像去畸变
        2. 对齐 lidar 和 camera 抽帧结果
    ros_bag_extractor.py 仅做保留，不建议使用
```


## Usage
```
    python rosbag_extractor_align_image_undistort.py --bag_path /mnt/data/20220122/data.bag --save_path /mnt/data/20220122/
```
默认会保存 camera-30, camera-60, camera-120 每 3 帧数据，lidar 的所有数据(bin格式)和 lidar 与 camera 时间对齐后的数据.
* 如果需要修改抽帧间隔或保存格式，参考代码内部参数.

运行后会自动生成多个以 sensor_id 命名的文件夹，每个文件夹内部是对应的各帧数据，如:
```
        .
        ├── data.bag
        ├── deb_version.txt
        ├── f120
        │   ├── 1640316386.900000095.jpg
        │   └── 1640316387.200000000.jpg
        ├── f30
        │   ├── 1640316386.900000095.jpg
        │   └── 1640316387.200000000.jpg
        ├── f60
        │   ├── 1640316386.900000095.jpg
        │   └── 1640316387.200000000.jpg
        ├── lidar_undist
        │   ├── 1640316386.800670624.bin
        │   └── 1640316386.900591612.bin
        ├── lidar_undist
        │   ├── 1640316386.800591612.bin
        │   └── 1640316386.900591612.bin
        ├── lidar_undist_align
        │   ├── 1640316386.900591612.bin
        └── └── 1640316387.200591612.bin

```


## Problems & Solutions

1、使用 python3 的问题:
```
    报错内容：
        ImportError: dynamic module does not define module export function (PyInit_cv_bridge_boost)

    问题原因: 
        使用 python3 运行该代码，需要在 python3 下编译 cv_bridge

    解决方法：
        参考：https://blog.csdn.net/qq_40297851/article/details/114396439
```

2、pcl 库的问题:
```
    报错内容：
        ImportError: AttributeError: 'module' object has no attribute 'PointCloud_PointXYZI'

    问题原因: 
        需要安装 python-pcl 库，使用源码安装

    解决方法：
        git clone git@github.com:strawlab/python-pcl.git
        cd python-pcl
        python setup.py install (--user)
        
        这里如果安装不通过，可尝试：
            1、修改 setup.py 中的 vtk 相关代码(想办法注释掉，参考 https://zhuanlan.zhihu.com/p/336875349)
            2、直接拷贝 build/lib.linux-x86_64-2.7/pcl 目录到 /home/zhanghao/.local/lib/python2.7/site-packages/ 下面
```
