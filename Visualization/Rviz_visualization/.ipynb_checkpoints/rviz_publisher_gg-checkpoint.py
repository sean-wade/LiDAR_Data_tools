#!/usr/bin/python3
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
        self.img_pub = rospy.Publisher('my_image', Image, queue_size=10)
        self.pcl_pub = rospy.Publisher('my_point_cloud', PointCloud2, queue_size=10)
        self.box_pub = rospy.Publisher('my_box3d', MarkerArray, queue_size=1)
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
                print(point_cloud.shape)
                publish_point_cloud(self.pcl_pub, point_cloud, down_ratio=3)

            if self.box_loader is not None:
                box_infos = self.box_loader.load_idx(frame_id)
                box_infos.filter_by_points_nums(5)
                pose = None
                if self.pose_loader is not None:
                    pose = self.pose_loader.load_idx(frame_id)
                publish_3dbox(self.box_pub, box_infos, pose, 1.0/self.ros_rate)

            frame_id += 1
            self.rate.sleep()


if __name__ == "__main__":
    import glob

    PUBLISH_SG_DATA = False
    if PUBLISH_SG_DATA:
        # 若使用 SG 包含 intensity 的 pcd, 速度会很慢, 考虑提前将 pcd 转为 npy/bin
        lidar_path = "/mnt/data/SGData/20220120/2022-01-20-10-02-30-morning_car_person/lidar_npy/*"
        image_path = "/mnt/data/SGData/20220120/2022-01-20-10-02-30-morning_car_person/cam60/*"
        annos_path = "/mnt/data/SGData/20220120/2022-01-20-10-02-30-morning_car_person/dets/*"

        lidar_files = sorted(glob.glob(lidar_path), key=lambda x: float(x[len(lidar_path) - 1:len(lidar_path) + 16]))
        image_files = sorted(glob.glob(image_path), key=lambda x: float(x[len(image_path) - 1:len(image_path) + 16]))
        annos_files = sorted(glob.glob(annos_path), key=lambda x: float(x[len(annos_path) - 1:len(image_path) + 16]))

        lidar_generator = LidarDataLoader(lidar_files, ndim=4)
        # lidar_generator = LidarDataLoader(lidar_files, ndim=3)
        image_generator = ImageDataLoader(image_files)
        annos_generator = Box3dDataLoader(annos_files, is_track=False, is_waymo=False)
        rp = RvizPublisher("sg_node", 30, lidar_generator, image_generator, annos_generator, field_angle=60)

        rp.Process()
    
    else:
        # Pandaset 生成的 det3d 数据
        # lidar_path = "/mnt/data/PandaSet/Det3d/lidar_all/"
        # lidar_path = "/mnt/data/PandaSet/Det3d/lidar_pandar64_inrange/"
        lidar_path = "/mnt/data/PandaSet/Det3d/lidar_pandarGT/"
        # lidar_path = "/mnt/data/PandaSet/Det3d/lidar_pandar64_inrange/"
        
        tokens = []
        for ii in range(1, 100):
            seq_ii_num = 80
            tokens += ["seq_%d_frame_"%ii + str(jj) for jj in range(seq_ii_num)]
            
        lidar_files = [lidar_path + tk + ".bin" for tk in tokens]
        lidar_generator = LidarDataLoader(lidar_files, ndim=4)
        
        if 0:
            annos_path = "/mnt/data/PandaSet/Det3d/annos_64_inrange/"
            anno_files = [annos_path + tk + ".txt" for tk in tokens]
            annos_generator = Box3dDataLoader(anno_files, filt_thres=0.1, is_track=True, is_waymo=True)
        else:
            annos_path = "/mnt/data/PandaSet/Det3d/preds_pandar64_inrange/"
            anno_files = [annos_path + tk + ".pkl" for tk in tokens]
            annos_generator = Box3dDataLoader(anno_files, filt_thres=0.3, is_track=False, is_waymo=False)

        

        
        
        # poses_generator = PoseLoader(pose_files)
        rp = RvizPublisher("sg_node", 1, lidar_generator, None, annos_generator, None, 0)

        rp.Process()


