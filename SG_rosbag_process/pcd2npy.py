#!/usr/bin/python2.7
'''
python2 pcd2npy.py \
    --pcd_path /mnt/data/SGData/20220120/2022-01-20-10-02-30-morning_car_person/lidar \
    --npy_path /mnt/data/SGData/20220120/2022-01-20-10-02-30-morning_car_person/lidar_npy
'''
import os
import pcl
import argparse
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def pcd2npy_folder(pcd_path, npy_path):
    pcds = os.listdir(pcd_path)
    if not os.path.exists(npy_path):
        os.makedirs(npy_path)

    for pcd in tqdm(pcds):
        points = pcl.load_XYZI(os.path.join(pcd_path, pcd)).to_array().astype(np.float32)
        # np.save(os.path.join(npy_path, pcd), points)
        points.tofile(os.path.join(npy_path, pcd))


def pcd2npy(paths):
    points = pcl.load_XYZI(paths[0]).to_array().astype(np.float32)
    # np.save(paths[1], points)
    points.tofile(paths[1])
    # print(points.dtype)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the MOTA & other metrics")

    parser.add_argument("--pcd_path", help="the dir to pcds", type = str, default = "./lidar")
    parser.add_argument("--npy_path", help="the dir to save npys",type = str, default = "./lidar_npy/")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # ## single-thread
    # pcd2npy_folder(args.pcd_path, args.npy_path)

    # ## multi-thread
    pcd_files = os.listdir(args.pcd_path)
    pcd_files = [os.path.join(args.pcd_path, pcd) for pcd in pcd_files]

    if not os.path.exists(args.npy_path):
        os.makedirs(args.npy_path)
    npy_files = [pcd.replace(args.pcd_path, args.npy_path)[:-4] + ".npy" for pcd in pcd_files]

    paths = zip(pcd_files, npy_files)

    pool = Pool(processes = 64)
    pool.map(pcd2npy, paths)
    pool.close()
    pool.join()

