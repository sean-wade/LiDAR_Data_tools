#!/usr/bin/python2
# -*- coding:utf-8 -*-

from lidar_loader import LidarDataLoader
from image_loader import ImageDataLoader
from box3d_loader import Box3dDataLoader
from poses_loader import PoseLoader
from publish_utils import *


class RvizPublisher(object):
    def __init__(self,
                 node_name   = "my_node", 
                 ros_rate    = 10,
                 pc_loader   = None,
                 img_loader  = None,
                 box_loader  = None,
                 pose_loader = None,
                 field_angle = 0,
                 ):
        self.ros_rate    = ros_rate
        self.pc_loader   = pc_loader
        self.img_loader  = img_loader
        self.box_loader  = box_loader
        self.pose_loader = pose_loader
        self.field_angle = field_angle
        
        rospy.init_node(node_name, anonymous=True)
        # self.img_pub = rospy.Publisher('my_image', Image, queue_size=10)
        self.pcl_pub = rospy.Publisher('my_point_cloud', PointCloud2, queue_size=10)
        self.box_pub = rospy.Publisher('my_box3d', MarkerArray, queue_size=10)
        self.ego_pub = rospy.Publisher('my_ego_view', Marker, queue_size=10)

        self.rate = rospy.Rate(self.ros_rate)
        self.bridge = CvBridge()


    def Process(self, startfrom=0):
        frame_id = startfrom
        while not rospy.is_shutdown():

            # Publish camera-image
            if self.img_loader is not None:
                image = self.img_loader.load_idx(frame_id)
                # publish_camera(self.img_pub, self.bridge, image)
                image = cv2.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3)))
                cv2.imshow("image", image)
                cv2.waitKey(1)
            
            # Publish camera field-angle 
            if self.field_angle > 0:
                publish_ego_car(self.ego_pub, self.field_angle)

            # # Publish point-cloud
            if self.pc_loader is not None:
                point_cloud = self.pc_loader.load_idx(frame_id)
                # print(point_cloud.shape)
                publish_point_cloud(self.pcl_pub, point_cloud, down_ratio=1)

            if self.box_loader is not None:
                box_infos = self.box_loader.load_idx(frame_id)
                pose = None
                if self.pose_loader is not None:
                    pose = self.pose_loader.load_idx(frame_id)
                # print(box_infos.label_ids)
                publish_3dbox(self.box_pub, box_infos, pose, 1.0/self.ros_rate)

            frame_id += 1
            # rospy.loginfo("published [%d]"%(frame_id))
            self.rate.sleep()


if __name__ == "__main__":
    import glob

    lidar_path = "/mnt/data/nuScenes-v1.0-mini/val_sep/points/"
    annos_path = "/mnt/data/nuScenes-v1.0-mini/val_sep/tk/"
    
    # poses_path = "/mnt/data/waymo_opensets/val/annos/"

    tokens = [str(i) for i in range(81)]

    lidar_files = [lidar_path + tk + ".bin" for tk in tokens]
    # pose_files = [poses_path + tk + ".txt" for tk in tokens]
    anno_files = [annos_path + tk + ".txt" for tk in tokens]

    lidar_generator = LidarDataLoader(lidar_files, ndim=4)
    annos_generator = Box3dDataLoader(anno_files, filt_thres=0.1, is_track=True)
    # annos_generator = Box3dDataLoader(anno_files, filt_thres=0.5, is_track=False)
    # poses_generator = PoseLoader(pose_files)
    # rp = RvizPublisher("sg_node", 10, lidar_generator, image_generator, annos_generator, poses_generator, 0)
    rp = RvizPublisher("sg_node", 1, lidar_generator, None, None, None, 0)

    rp.Process(startfrom = 0)


