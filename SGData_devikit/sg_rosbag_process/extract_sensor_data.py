#!/usr/bin/python2.7

import os
import sys

import argparse
from tqdm import tqdm

import rosbag
from cv_bridge import CvBridge

from cfg import default_topic_sensor_name_dict, default_calibration_files
from utils import logger, save_image_msg, save_lidar_msg, save_imu_msg, save_ins_msg, check_folder_file_consistant, load_intrin_and_coef


class BagExtractor(object):
    def __init__(self, rosbag_path, save_path, lidar_save_pcd=False):
        self.rosbag_path = rosbag_path
        self.save_path = save_path
        self.lidar_save_pcd = lidar_save_pcd
        
        if not os.path.exists(rosbag_path):
            logger.critical("rosbag [%s] doesn't exist !!!"%rosbag_path)
            sys.exit(-1)

        if os.path.exists(self.save_path):
            logger.critical("Save path already exists, plz check [%s]!"%self.save_path)
            sys.exit(-1)

        self.lidar_topics = list(default_topic_sensor_name_dict["lidar"].keys())
        logger.info("Lidar topics : %s"%self.lidar_topics)

        self.ins_topics = list(default_topic_sensor_name_dict["ins"].keys())
        logger.info("INS topics : %s"%self.ins_topics)

        self.imu_topics = list(default_topic_sensor_name_dict["imu"].keys())
        logger.info("IMU topics : %s"%self.imu_topics)

        self.bridge = CvBridge()
        self.camera_topics = list(default_topic_sensor_name_dict["camera"].keys())
        logger.info("Cam topics : %s"%self.camera_topics)
        self.camera_calibs = {
            k : load_intrin_and_coef(default_calibration_files.get(k)) for k in default_topic_sensor_name_dict["camera"].keys()
        }
        logger.info("Cam calibs : \n%s"%self.camera_calibs)
        
        for cam_topic in self.camera_topics:
            if self.camera_calibs.get(cam_topic)[0] is not None:
                default_topic_sensor_name_dict["camera"][cam_topic] += "_undist"

        self.sensor_folder_names = []
        for _, cur_sensors in default_topic_sensor_name_dict.items():
            self.sensor_folder_names.extend(list(cur_sensors.values()))

        for sensor_name in self.sensor_folder_names:
            sensor_folder = os.path.join(self.save_path, sensor_name)
            if not os.path.exists(sensor_folder):
                os.makedirs(sensor_folder)
                logger.info("Created sensor save folder : [%s]"%sensor_folder)


    def save_all_to_disk(self):
        try:
            with rosbag.Bag(self.rosbag_path, 'r') as bag:
                logger.info("Loading bag success : [%s]"%self.rosbag_path)

                for topic, src_msg, t in tqdm(bag.read_messages()):
                    if topic in self.lidar_topics:
                        ts = src_msg.header.stamp.to_sec()
                        save_name = "%.9f.pcd" % ts if self.lidar_save_pcd else "%.9f.bin" % ts
                        save_path = os.path.join(os.path.join(self.save_path, default_topic_sensor_name_dict["lidar"][topic]), save_name)
                        save_lidar_msg(src_msg, save_path, self.lidar_save_pcd)
                    
                    elif topic in self.camera_topics:
                        intrinsics = self.camera_calibs[topic][0]
                        distcoeffs = self.camera_calibs[topic][1]
                        img_name = "%.9f.jpg" % src_msg.header.stamp.to_sec()
                        img_path = os.path.join(os.path.join(self.save_path, default_topic_sensor_name_dict["camera"][topic]), img_name)
                        save_image_msg(src_msg, self.bridge, img_path, intrinsics, distcoeffs)

                    elif topic in self.ins_topics:
                        ins_name = "%.9f.yaml" % src_msg.header.stamp.to_sec()
                        ins_path = os.path.join(os.path.join(self.save_path, default_topic_sensor_name_dict["ins"][topic]), ins_name)
                        save_ins_msg(src_msg, ins_path)

                    elif topic in self.imu_topics:
                        imu_name = "%.9f.yaml" % src_msg.header.stamp.to_sec()
                        imu_path = os.path.join(os.path.join(self.save_path, default_topic_sensor_name_dict["imu"][topic]), imu_name)
                        save_imu_msg(src_msg, imu_path)

        except Exception as e:
            logger.critical(str(e))
            sys.exit(-1)

    
    def check_consistant(self):
        all_consistant = True
        for sensor_name in self.sensor_folder_names:
            if sensor_name in ["ins", "imu"]:
                hz = 1.0 / 100
            else:
                hz = 1.0 / 10.0
            sensor_folder = os.path.join(self.save_path, sensor_name)
            all_consistant = check_folder_file_consistant(sensor_folder, hz)
        return all_consistant


    def save_and_check(self):
        self.save_all_to_disk()
        logger.info("Success save all data to disk, see [%s]"%self.save_path)

        is_consistant = self.check_consistant()
        if is_consistant:
            logger.info("Check consistant passed, all data is continuous.")
        else:
            logger.error("Check consistant failed, please check in folder : [%s]"%self.save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Split ros_bag to seperate files.")

    parser.add_argument("--bag_path", help="the path to rosbag", type = str, default="")
    parser.add_argument("--save_path", help="the path to save data",type = str, default = "")
    parser.add_argument("--save_pcd", help="save bin instead of pcd", action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    bag_extractor = BagExtractor(args.bag_path, args.save_path, lidar_save_pcd=args.save_pcd)                 
    bag_extractor.save_and_check()  
