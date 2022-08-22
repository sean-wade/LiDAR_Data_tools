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
                   "f30", 
                   "f60", 
                   "f120", 
                   #"fl120", 
                   #"fr120", 
                   #"rl120", 
                   #"rr120", 
                   #"r120"
                   ]

Default_lidar_topics = {
    #"/sensor/lidar/fusion" : "lidar",
    "/sensor/lidar/undistortion" : "lidar_undist",
}
# default topic name
Default_camera_topics = {
    "f30"    : "/sensor/camera/f30",
    "f60"    : "/sensor/camera/f60",
    "f120"   : "/sensor/camera/f120",
    #"fl120"  : "/sensor/camera/fl120",
    #"fr120"  : "/sensor/camera/fr120",
    #"rl120"  : "/sensor/camera/rl120",
    #"rr120"  : "/sensor/camera/rr120",
    #"r120"   : "/sensor/camera/r120",
}


class BagExtractor(object):
    def __init__(self, rosbag_path, extract_path, compressed=True):
        self.rosbag_path = rosbag_path
        self.extract_path = extract_path
        self.compressed = compressed
        assert os.path.exists(rosbag_path), "rosbag [%s] doesn't exist !!!"%rosbag_path


    def ExtractLidar(self):
        lidar_folder = os.path.join(self.extract_path, "lidar")
        lidar_folder_undist = os.path.join(self.extract_path, "lidar_undist")
        #lidar_folder_npy = os.path.join(self.extract_path, "lidar_npy")

        if not os.path.exists(lidar_folder):
            os.makedirs(lidar_folder)

        #if not os.path.exists(lidar_folder_npy):
        #    os.makedirs(lidar_folder_npy)

        if not os.path.exists(lidar_folder_undist):
            os.makedirs(lidar_folder_undist)
        
        topic_name = Default_lidar_topics.keys()[0]
        try:
            with rosbag.Bag(self.rosbag_path, 'r') as bag:
                print("Loading bag success : ", self.rosbag_path)

                for topic, src_msg, t in bag.read_messages():
                    # print(topic)
                    if topic == topic_name:
                        src_points = pc2.read_points_list(src_msg, 
                            field_names=( "x", "y", "z", "intensity"), skip_nans=True)
                        # print(src_points)
                        pc = pcl.PointCloud_PointXYZI(src_points)
                        pcd_name = "%.9f.pcd" % src_msg.header.stamp.to_sec()
                        pcd_path = os.path.join(lidar_folder, pcd_name)
                        print(pcd_path)
                        pcl.save(pc, pcd_path)

                        npy_path = os.path.join(lidar_folder_npy, pcd_name)
                        np.save(npy_path, pc.to_array())
        finally:
            bag.close()

    
    def ExtractImage(self, camera_name="f30"):
        img_folder = os.path.join(self.extract_path, camera_name)
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)

        topic_name = Default_camera_topics[camera_name]
        if self.compressed:
            topic_name = topic_name + "/compressed"

        bridge = CvBridge()
        try:
            with rosbag.Bag(self.rosbag_path, 'r') as bag:
                print("Loading bag success : ", self.rosbag_path)

                for topic, src_msg, t in bag.read_messages():
                    if topic == topic_name:
                        if self.compressed:
                            cv_image = bridge.compressed_imgmsg_to_cv2(src_msg, "bgr8")
                        else:
                            cv_image = bridge.imgmsg_to_cv2(src_msg, "bgr8")
                        img_name = "%.9f.jpg" % src_msg.header.stamp.to_sec()
                        img_path = os.path.join(img_folder, img_name)
                        print(img_path)
                        cv2.imwrite(img_path, cv_image)
        finally:
            bag.close()


    def ExtractAll(self, interval=3):
        for sensor_name in Default_sensors:
            sensor_folder = os.path.join(self.extract_path, sensor_name)
            if not os.path.exists(sensor_folder):
                os.makedirs(sensor_folder)

        #lidar_folder_npy = os.path.join(self.extract_path, "lidar_npy")
        #if not os.path.exists(lidar_folder_npy):
        #    os.makedirs(lidar_folder_npy)

        bridge = CvBridge()
        camera_topics = list(Default_camera_topics.values())
        if self.compressed:
            camera_topics = [tpc + "/compressed" for tpc in camera_topics]
        print("Cam topics : ", camera_topics)
        
        lidar_topics = list(Default_lidar_topics.keys())
        print("Lidar topics : ", lidar_topics)
        
        topic_frame_count = {}
        try:
            with rosbag.Bag(self.rosbag_path, 'r') as bag:
                print("Loading bag success : ", self.rosbag_path)

                for topic, src_msg, t in bag.read_messages():
                    # print(topic)
                    if topic in topic_frame_count:
                        topic_frame_count[topic] += 1
                    else:
                        topic_frame_count.update({topic : 1})
                    if topic_frame_count[topic] % interval != 0:
                        continue
                        
                    if topic in lidar_topics:
                        src_points = pc2.read_points_list(src_msg, 
                            field_names=( "x", "y", "z", "intensity"), skip_nans=True)
                            
                        pc = pcl.PointCloud_PointXYZI(src_points)
                        pcd_name = "%.9f.pcd" % src_msg.header.stamp.to_sec()
                        pcd_path = os.path.join(os.path.join(self.extract_path, Default_lidar_topics[topic]), pcd_name)
                        print(pcd_path)
                        pcl.save(pc, pcd_path)

                        #npy_path = os.path.join(lidar_folder_npy, pcd_name)
                        #np.save(npy_path, pc.to_array())
                    
                    elif topic in camera_topics:
                        if self.compressed:
                            cv_image = bridge.compressed_imgmsg_to_cv2(src_msg, "bgr8")
                        else:
                            cv_image = bridge.imgmsg_to_cv2(src_msg, "bgr8")
                        img_name = "%.9f.jpg" % src_msg.header.stamp.to_sec()
                        img_path = os.path.join(os.path.join(self.extract_path, topic.split("/")[3]), img_name)
                        print(img_path)
                        cv2.imwrite(img_path, cv_image)

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
    bag_extractor.ExtractAll(interval=3)                             

