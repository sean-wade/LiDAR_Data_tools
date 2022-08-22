#!/usr/bin/python2.7
import os
import argparse
from ros_bag_extractor import BagExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Split ros_bag to seperate files.")

    parser.add_argument("--folder", help="the dir to groundtruth", type = str, default = "/mnt/Public/Rosbag/20220120")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    assert os.path.exists(args.folder), "folder [%s] doesn't exist !!!"%args.folder

    sub_folders = os.listdir(args.folder)
    
    for sub_folder in sub_folders:
        
        bag_path = args.folder + "/" + sub_folder + "/data_with_undistortion_2.bag"
        if not os.path.exists(bag_path):
            continue
        
        save_path = "/mnt/data/SGData/20220120_for_annotate/" + sub_folder + "/for_annotate"
        if os.path.exists(save_path):
            os.system("rm -rf %s"%save_path)
        os.makedirs(save_path)
        
        bag_extractor = BagExtractor(bag_path, save_path)                         
        bag_extractor.ExtractAll(interval=3)                             

