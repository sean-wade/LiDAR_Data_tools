#!/usr/bin/python2.7

import os
import cv2
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import rospy
import rosbag
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2


try:
    import pcl
except Exception as e:
    print(e)
    print("If pcl library has a problem, plz save as numpy-bin file !")

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

# set intrinsics matrics and distcoeffs matrics
intrinsics_f60 = np.array([[1.9641e3, 1.0997, 995.8989],
                           [0, 1.9643e3, 557.9326],
                           [0, 0, 1]])
distcoeffs_f60 = np.array([-0.5638, 0.3916, 0.0006, -0.0035, -0.2709])

intrinsics_f120 = np.array([[1.0109e3, 1.0997, 995.8989],
                            [0, 1.0129e3, 559.3838],
                            [0, 0, 1]])
distcoeffs_f120 = np.array([-0.3714, 0.1674, -8.1010e-04, -9.5426e-05, -0.0409])


# TODO: 请先检查 lidar 与 camera folder 中的文件的时间戳，是否全部连续，若不连续，需报错退出！
def align_folder(cam_path, lidar_path, lidar_out_path, ts_diff=0.01):
    print("Start align lidar-folder[%s] to camera-folder:[%s]"%(lidar_path, cam_path))
    camera_name_list = os.listdir(cam_path)
    lidar_name_list = os.listdir(lidar_path)
    os.makedirs(lidar_out_path)
    os.system("rm %s/*.pcd"%lidar_out_path)

    for cam_name in tqdm(camera_name_list):
        for lidar_name in lidar_name_list:
            cam_ts = float(cam_name[:-4])
            lidar_ts = float(lidar_name[:-4])
            if abs(cam_ts - lidar_ts) < ts_diff:
                lidar_from_path = os.path.join(lidar_path, lidar_name)
                lidar_to_path = os.path.join(lidar_out_path, lidar_name)
                shutil.copy(lidar_from_path, lidar_to_path)
    print("Finished align, new lidar-folder is ", lidar_out_path)


class BagExtractor(object):
    def __init__(self, rosbag_path, extract_path, compressed=True, lidar_save_pcd=True):
        self.rosbag_path = rosbag_path
        self.extract_path = extract_path
        self.compressed = compressed
        self.lidar_save_pcd = lidar_save_pcd
        assert os.path.exists(rosbag_path), "rosbag [%s] doesn't exist !!!"%rosbag_path


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
                # print("Message_count :\n", bag.get_message_count())

                for topic, src_msg, t in tqdm(bag.read_messages()):
                    # print(topic)
                    if topic in topic_frame_count:
                        topic_frame_count[topic] += 1
                    else:
                        topic_frame_count.update({topic : 1})
                        
                    if topic in lidar_topics:
                        src_points = pc2.read_points_list(src_msg, 
                            field_names=( "x", "y", "z", "intensity"), skip_nans=True)

                        if self.lidar_save_pcd:
                            pc = pcl.PointCloud_PointXYZI(src_points)
                            pcd_name = "%.9f.pcd" % src_msg.header.stamp.to_sec()
                            pcd_path = os.path.join(os.path.join(self.extract_path, Default_lidar_topics[topic]), pcd_name)
                            # print(pcd_path)
                            pcl.save(pc, pcd_path)
                        else:
                            points = np.array(src_points, dtype = np.float32)
                            pcd_name = "%.9f.bin" % src_msg.header.stamp.to_sec()
                            pcd_path = os.path.join(os.path.join(self.extract_path, Default_lidar_topics[topic]), pcd_name)
                            points.tofile(pcd_path)

                        #npy_path = os.path.join(lidar_folder_npy, pcd_name)
                        #np.save(npy_path, pc.to_array())
                    
                    elif topic in camera_topics:
                        if topic_frame_count[topic] % interval == 0:
                            if self.compressed:
                                cv_image = bridge.compressed_imgmsg_to_cv2(src_msg, "bgr8")
                            else:
                                cv_image = bridge.imgmsg_to_cv2(src_msg, "bgr8")
                            
                            # image undistort
                            image_undistort = cv_image.copy()
                            if topic == "/sensor/camera/f120/compressed":
                                cv2.undistort(cv_image, intrinsics_f120, distcoeffs_f120, image_undistort, None)
                            if topic == "/sensor/camera/f60/compressed":
                                cv2.undistort(cv_image, intrinsics_f60, distcoeffs_f60, image_undistort, None)
                            
                            img_name = "%.9f.jpg" % src_msg.header.stamp.to_sec()
                            img_path = os.path.join(os.path.join(self.extract_path, topic.split("/")[3]), img_name)
                            # print(img_path)
                            cv2.imwrite(img_path, image_undistort)
            
            align_folder(
                os.path.join(self.extract_path, Default_camera_topics.keys()[0]),
                os.path.join(self.extract_path, Default_lidar_topics.values()[0]),
                os.path.join(self.extract_path, Default_lidar_topics.values()[0] + "_align"),
            )

        finally:
            bag.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Split ros_bag to seperate files.")

    parser.add_argument("--bag_path", help="the dir to groundtruth", type = str, default = "./data.bag")
    parser.add_argument("--save_path", help="the path to gt infos",type = str, default = "./data/")
    parser.add_argument("--save_pcd", help="save bin instead of pcd", action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    bag_extractor = BagExtractor(args.bag_path, args.save_path, lidar_save_pcd=args.save_pcd)                 
    bag_extractor.ExtractAll(interval=3)                             


