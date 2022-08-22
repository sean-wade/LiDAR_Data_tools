#!/usr/bin/python2
# -*- coding:utf-8 -*-

import pcl
import pickle
import numpy as np
# import open3d as o3d

# from publish_utils import *


def load_lidar_npy(npy_path, ndim=3):
    """
    Args:
        npy_path (str): .npy or .bin file
        ndim (int, optional): [3 + pointcloud features]. Defaults to 3.
    """
    points = np.fromfile(npy_path, dtype=np.float32).reshape((-1, ndim))
    return points


def load_lidar_txt(txt_path):
    points = np.loadtxt(txt_path)
    return points


def load_lidar_pkl(pkl_path):
    """
    Args:
        pkl_path (str): pkl file_path which is generated from det3d framework
    """
    data = pickle.load(open(pkl_path, 'rb'))
    xyz = data['lidars']['points_xyz']
    intensity = data['lidars']['points_feature'][:, 0].reshape(-1,1)
    
    return np.hstack((xyz, intensity))


# def load_lidar_pcd(pcd_path, ndim=3):
#     """
#     Args:
#         pcd_path (str): pcd path from SG rosbag or other
#         padding (int, optional): [how many columns to pad]. Defaults to 0.
#     Attention:
#         Intensity maybe in pcd-file, this way will ignore it
#         (Should use other way to load pcd).
#     """
#     pcd = o3d.io.read_point_cloud(pcd_path)
#     pcd_npy = np.asarray(pcd.points)
#     padding = ndim - 3
#     if padding > 0:
#         pcd_npy = np.append(pcd_npy, np.zeros((pcd_npy.shape[0], padding)), axis=1)
#     return pcd_npy


def load_lidar_pcd(pcd_path, ndim=3):
    """
    Comments:
        If pcd has intensity, will be very slow.(May be SG-data's reason)
    """
    if ndim == 3:
        points = pcl.load(pcd_path).to_array()
        return points

    elif ndim == 4:
        points = pcl.load_XYZI(pcd_path).to_array()
        return points
    
    else:
        print("Unsurpported ndim: ", ndim)
        return np.empty(0)
    

class LidarDataLoader(object):
    def __init__(self, lidar_files, ndim=3):
        """[summary]

        Args:
            lidar_files (list): [list of full_path with .npy/.bin/.pkl/.pcd/.txt extension files]
            # ! < MUST SORT > !
        """
        self.idx = 0
        self.ndim = ndim
        self.lidar_files = lidar_files
        self.lidar_nums = len(self.lidar_files)
        print("LidarDataLoader Init with [%d] files"%self.lidar_nums)

    
    def __len__(self):
        return self.lidar_nums


    def __iter__(self):
        return self

    
    def __next__(self):
        """[summary]
        Returns:
            np.array: (3 + F) * N
        """
        if self.idx == self.lidar_nums:
            raise StopIteration

        lidar_path = self.lidar_files[self.idx]
        file_type = lidar_path.split(".")[-1]
        points = self.get_data(lidar_path, file_type)

        self.idx += 1
        return points


    def load_idx(self, idx):
        if idx >= self.lidar_nums:
            print("Load lidar index[%d] out of range[%d] !!!"%(idx, self.lidar_nums))
            return None

        lidar_path = self.lidar_files[idx]
        file_type = lidar_path.split(".")[-1]
        points = self.get_data(lidar_path, file_type)
        return points

    
    def get_data(self, lidar_path, file_type):
        print("lidar_path: ", lidar_path)
        points = np.empty(0)
        
        if file_type == "npy" or file_type == "bin":
            points = load_lidar_npy(lidar_path, self.ndim)
        elif file_type == "pcd":
            points = load_lidar_pcd(lidar_path, self.ndim)            
        elif file_type == "txt":
            points = load_lidar_txt(lidar_path)
        elif file_type == "pkl":
            points = load_lidar_pkl(lidar_path)
        else:
            print("File extension cannot understand, return empty array : ", lidar_path)
        
        return points



if __name__ == "__main__":
    import os
    import glob

    # ! only test load, unsorted files.
    # files = glob.glob("/mnt/data/SGData/20220120/2022-01-20-09-44-33-morning/lidar/*.pcd")
    files = glob.glob("/mnt/data/waymo_opensets/val/lidar/*")
    # files = glob.glob("/mnt/data/waymo_opensets/train/lidar/*")
    # files = glob.glob("./lidar/*")
    lidar_generator = LidarDataLoader(files)

    print("Total lidar files = ", len(lidar_generator))
    # for points in lidar_generator:
    #     print(points.shape)

    iii = 0
    while True:
        # points = next(lidar_generator)
        # points = lidar_generator.__next__()
        points = lidar_generator.load_idx(iii)
        print(points.shape)
        iii += 1



