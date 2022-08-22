#!/usr/bin/python2
# -*- coding:utf-8 -*-
import os
import glob
import argparse

from lidar_loader import LidarDataLoader
from image_loader import ImageDataLoader
from box3d_loader import Box3dDataLoader
from poses_loader import PoseLoader
from rviz_publisher import RvizPublisher


def parse_args():
    parser = argparse.ArgumentParser(description="Rviz visualization project")

    parser.add_argument("--lidar_path", help="the dir to lidar files", type = str)
    parser.add_argument("--image_path", help="the dir to image files", type = str)
    parser.add_argument("--rate", help="the ros rate", type = int, default=10)
    parser.add_argument("--fov", help="the filed angle of camera", type = int, default=60)
    args = parser.parse_args()
    return args


def main(lidar_path, image_path, rate, fov):
    lidar_generator = image_generator = None

    if lidar_path and os.path.exists(lidar_path):
        lidar_files = sorted(glob.glob(os.path.join(lidar_path, "*")))
        print(lidar_files[:20])
        lidar_generator = LidarDataLoader(lidar_files, ndim=4)

    if image_path and os.path.exists(image_path):
        image_files = sorted(glob.glob(os.path.join(image_path, "*")))
        print(image_files[:20])
        image_generator = ImageDataLoader(image_files)

    if lidar_generator is None and image_generator is None:
        print("Lidar & Image path is invalid, please check !!!")
        return 
    elif len(lidar_generator)==0 and len(image_generator)==0:
        print("Lidar & Image folder is empty, please check !!!")
        return 
    else:
        rp = RvizPublisher("sg_node", rate, 
                        lidar_generator, 
                        image_generator, 
                        None, 
                        None,
                        fov)
        rp.Process()


if __name__ == "__main__":
    args = parse_args()
    main(args.lidar_path, 
         args.image_path,
         args.rate,
         args.fov,
         )

    """
    e.g.
    python main.py \
        --lidar_path /mnt/data/SGData/20220120/2022-01-20-09-52-09-morning_inv/lidar_npy \
        --image_path /mnt/data/SGData/20220120/2022-01-20-09-52-09-morning_inv/cam60 \
        --rate 50 \
        --fov 60
    """