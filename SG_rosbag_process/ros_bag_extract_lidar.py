#!/usr/bin/python2.7

import os
import cv2
import pcl
import argparse
import numpy as np

import rospy
import rosbag
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2


# default 9 sensor names
Default_sensors = [
                   # "lidar", 
                   "lidar_undist", 
                   ]

Default_lidar_topics = {
    #"/sensor/lidar/fusion" : "lidar",
    "/sensor/lidar/undistortion" : "lidar_undist",
}



class BagExtractor(object):
    def __init__(self, rosbag_path, extract_path, compressed=True):
        self.rosbag_path = rosbag_path
        self.extract_path = extract_path
        self.compressed = compressed
        assert os.path.exists(rosbag_path), "rosbag [%s] doesn't exist !!!"%rosbag_path


    def ExtractLidar(self, interval=3, shift=2):
        for sensor_name in Default_sensors:
            sensor_folder = os.path.join(self.extract_path, sensor_name)
            if not os.path.exists(sensor_folder):
                os.makedirs(sensor_folder)
        
        lidar_topics = list(Default_lidar_topics.keys())
        print("Lidar topics : ", lidar_topics)
        
        topic_frame_count = {}
        try:
            with rosbag.Bag(self.rosbag_path, 'r') as bag:
                print("Loading bag success : ", self.rosbag_path)
                
                frame_num = 0
                for topic, src_msg, t in bag.read_messages():
                    # print(topic)
                    if topic in topic_frame_count:
                        topic_frame_count[topic] += 1
                    else:
                        topic_frame_count.update({topic : 1})
                    if topic_frame_count[topic] % interval == shift:
                        if topic in lidar_topics:
                            src_points = pc2.read_points_list(src_msg, 
                                field_names=( "x", "y", "z", "intensity"), skip_nans=True)
                                
                            pc = pcl.PointCloud_PointXYZI(src_points)
                            pcd_name = "%.9f.pcd" % src_msg.header.stamp.to_sec()
                            # pcd_name = "seq_0_frame_%d.pcd" % frame_num
                            pcd_path = os.path.join(os.path.join(self.extract_path, Default_lidar_topics[topic]), pcd_name)
                            print(pcd_path)
                            pcl.save(pc, pcd_path)
                            frame_num += 1

                    else:
                        continue
        finally:
            bag.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Split ros_bag to seperate files.")

    parser.add_argument("--bag_path", help="the dir to groundtruth", type = str, default = ".data.bag")
    parser.add_argument("--save_path", help="the path to gt infos",type = str, default = "./data/")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # bag_extractor = BagExtractor("/mnt/data/SGData/20211224/2021-12-24-11-26-24-2/data.bag",
    #                              "/mnt/data/SGData/20211224/2021-12-24-11-26-24-2/")

    bag_extractor = BagExtractor(args.bag_path, args.save_path)

    # bag_extractor.ExtractLidar()                             
    # bag_extractor.ExtractImage("f60")                             
    bag_extractor.ExtractLidar(interval=1, shift=0)                             

