import os
import sys
import cv2
import shutil
import argparse
import numpy as np
from tqdm import tqdm

from cfg import default_topic_sensor_name_dict
from utils import logger, check_folder_file_consistant


def parse_args():
    parser = argparse.ArgumentParser(description="Split ros_bag to seperate files.")

    parser.add_argument("--save_path", help="the path to saved data",type = str, default = "")
    parser.add_argument("--interval", help="the interval of extract",type = int, default = 3)
    args = parser.parse_args()
    return args


def extract(src_dir, dst_dir, select_times, ts_diff=0.01):
    logger.info("Start extract folder [%s]."%(src_dir))
    f_names = os.listdir(src_dir)
    not_found_num = 0
    found_num = 0
    for select_ts in (select_times):
        found = False
        for f_name in f_names:
            cur_ts = float(f_name[:-5])
            if abs(cur_ts - select_ts) < ts_diff:
                from_path = os.path.join(src_dir, f_name)
                to_path = os.path.join(dst_dir, f_name)
                shutil.copy(from_path, to_path)
                found = True
                found_num += 1
                break
                
        if not found:
            logger.error("Cannot find timestamp ~ [%.9f] file in [%s]!!!"%(select_ts, src_dir))
            not_found_num += 1
    return not_found_num, found_num


def select_by_intervel(data_path, interval):
    sensor_folder_names = os.listdir(data_path)
    # for _, cur_sensors in default_topic_sensor_name_dict.items():
    #     sensor_folder_names.extend(list(cur_sensors.values()))

    select_path = os.path.join(data_path, "select")
    if os.path.exists(select_path):
        logger.error("Select save path already exists, plz check [%s]!"%select_path)
        sys.exit(-1)

    for sensor_name in sensor_folder_names:
        hz = 0.01 if ("ins" in sensor_name or "imu" in sensor_name) else 0.1
        check_folder_file_consistant(os.path.join(data_path, sensor_name), hz)
        sensor_folder = os.path.join(select_path, sensor_name)
        os.makedirs(sensor_folder)
    
    # select the first camera as main sensor.
    main_sensor = list(default_topic_sensor_name_dict["camera"].values())[0]
    all_file_names = os.listdir(os.path.join(data_path, main_sensor))
    all_file_names.sort()
    selected_tss = [float(x[:-4]) for x in all_file_names][::interval]
    
    for sensor_name in sensor_folder_names:
        src_dir = os.path.join(data_path, sensor_name)
        dst_dir = os.path.join(select_path, sensor_name)
        not_found_num, _ = extract(src_dir, dst_dir, selected_tss)
        if not_found_num > 0:
            logger.error("Sensor [%s] cannot find [%d] files ! Plz check !!!"%(sensor_name, not_found_num))

    # final check file numbers.
    final_nums = []
    for sensor_name in sensor_folder_names:
        sensor_data_nums = len(os.listdir(dst_dir))
        logger.info("After select,  [%d] files in [%s]."%(sensor_data_nums, sensor_name))
        dst_dir = os.path.join(select_path, sensor_name)
        final_nums.append(sensor_data_nums)

    if final_nums.count(final_nums[0]) == len(final_nums):
        logger.info("Extract finished, see [%s], each folder has [%d] files."%(select_path, final_nums[0]))
    else:
        logger.critical("Extract finished, but the file nums: %s are not equal !!! Plz check [%s]!!!"%(str(final_nums), select_path))




if __name__ == "__main__":
    args = parse_args()
    select_by_intervel(args.save_path, args.interval)
