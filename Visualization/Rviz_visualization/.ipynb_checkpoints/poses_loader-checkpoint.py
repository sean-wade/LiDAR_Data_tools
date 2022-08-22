#!/usr/bin/python2
# -*- coding:utf-8 -*-

import pickle
import numpy as np


class PoseLoader(object):
    def __init__(self, pose_files):
        """[summary]

        Args:
            pose_files (list): [list of full_path with .pkl/.txt extension files]
            # ! < MUST SORT > !
        """
        self.idx = 0
        self.pose_files = pose_files
        self.pose_nums = len(self.pose_files)
        print("PoseLoader Init with [%d] files"%self.pose_nums)

    
    def __len__(self):
        return self.pose_nums


    def __iter__(self):
        return self

    
    def __next__(self):
        """
        Returns:
            FrameBoxData
        """
        if self.idx == self.pose_nums:
            raise StopIteration

        pose_path = self.pose_files[self.idx]
        pose = self.get_data(pose_path)

        self.idx += 1
        return pose


    def load_idx(self, idx):
        if idx >= self.pose_nums:
            print("Load info index[%d] out of range[%d] !!!"%(idx, self.pose_nums))
            raise StopIteration

        pose_path = self.pose_files[idx]
        return self.get_data(pose_path)


    def get_data(self, pose_path):
        file_ext = pose_path.split(".")[-1]
        if file_ext == "txt":
            return np.loadtxt(pose_path).reshape(4, 4)
        elif file_ext == "pkl":
            dd = pickle.load(open(pose_path, "rb"))
            return dd["veh_to_global"].reshape(4, 4)
        else:
            return np.eye(4, 4)


if __name__ == "__main__":
    import os
    import glob

    files = []
    for ii in range(202):
        # files += sorted(glob.glob("/mnt/data/waymo_opensets/val/annos/seq_%d_frame_*"%ii), key=lambda x: int(x.split("/")[-1][:-4].split("_")[-1]))
        files += sorted(glob.glob("/mnt/data/waymo_opensets/val/for_cpp/poses/seq_%d_frame_*"%ii), key=lambda x: int(x.split("/")[-1][:-4].split("_")[-1]))

    pose_loader = PoseLoader(files)
    iii = 0
    while True:
        rrr = pose_loader.load_idx(iii)
        print(rrr)
        if iii == 2:
            break
        iii += 1




