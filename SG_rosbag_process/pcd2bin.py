import os
import pcl
import glob
import tqdm
import numpy as np


def load_pcd_pad(pcd_path, padding=0):
    points = pcl.load_XYZI(pcd_path).to_array()
    points[:, 3] = np.tanh(points[:, 3])
    
    if padding > 0:
        points = np.append(points, np.zeros((points.shape[0], padding)), axis=1)
    return points
        
pcds = glob.glob("/mnt/data/SGData/20220120/2022-01-20-14-02-36-afternoon_inv/lidar_undist/*.pcd")
pcds = sorted(pcds, key=lambda x: float(x.split("/")[-1][:-4]))


for pcd in tqdm.tqdm(pcds):
    points = load_pcd_pad(pcd, 1)
    bin_path = pcd.replace("lidar_undist", "bin")[:-4] + ".bin"
    points.tofile(bin_path)

