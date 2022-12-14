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
                publish_point_cloud(self.pcl_pub, point_cloud, 5)

            if self.box_loader is not None:
                box_infos = self.box_loader.load_idx(frame_id)
                pose = None
                if self.pose_loader is not None:
                    pose = self.pose_loader.load_idx(frame_id)
                # print(box_infos.label_ids)
                publish_3dbox(self.box_pub, box_infos, pose, 1.0/self.ros_rate)

            frame_id += 1
            # if frame_id > 1:
            #     frame_id = 0
            # rospy.loginfo("published [%d]"%(frame_id))
            self.rate.sleep()


if __name__ == "__main__":
    import glob

    PUBLISH_SG_DATA = 0
    if PUBLISH_SG_DATA:
        # ????????? SG ?????? intensity ??? pcd, ???????????????, ??????????????? pcd ?????? npy/bin
        lidar_path = "/mnt/data/SGTrain/20220801/lidar/lidar_undist/*"
        # image_path = "/mnt/Public_zk/Rosbag/20220120/2022-01-20-09-44-33-morning/for_annotate/f30/*"
        annos_path = "/home/zhanghao/code/GitLab/lidar_ped_veh_model/work_dirs/sg_trian1/eval_det_multi/inference/*"
        # annos_path = "/mnt/data/SGData/20220120/2022-01-20-10-02-30-morning_car_person/trks/*"

        lidar_files = sorted(glob.glob(lidar_path), key=lambda x: float(x[len(lidar_path) - 1:len(lidar_path) + 16]))
        # image_files = sorted(glob.glob(image_path), key=lambda x: float(x[len(image_path) - 1:len(image_path) + 16]))
        annos_files = sorted(glob.glob(annos_path), key=lambda x: float(x[len(annos_path) - 1:len(annos_path) + 16]))[1:]

        lidar_generator = LidarDataLoader(lidar_files, ndim=4)
        # lidar_generator = LidarDataLoader(lidar_files, ndim=3)
        # image_generator = ImageDataLoader(image_files)
        image_generator = None
        annos_generator = Box3dDataLoader(annos_files, is_track=False, is_waymo=True)
        # annos_generator = None
        rp = RvizPublisher("sg_node", 10, lidar_generator, image_generator, annos_generator, field_angle=60)
        # rp = RvizPublisher("sg_node", 10, lidar_generator, image_generator, None, field_angle=60)

        rp.Process()
    
#     else:
#         # ??? waymo ??????
#         lidar_path = "/home/zhanghao/code/GitLab/lidar_ped_veh_pipeline/lidar_perception/test_seq_0/bin/"
#         annos_path = "/home/zhanghao/code/GitLab/lidar_ped_veh_pipeline/lidar_perception/test_seq_0/api_trk_res2/"
        
#         tokens = []
#         select_seq = [0]
#         for ii in select_seq:
#             seq_ii_num = len(glob.glob(lidar_path + "/seq_%d_frame_*"%ii))
#             tokens += ["seq_%d_frame_"%ii + str(jj) for jj in range(seq_ii_num)]

#         lidar_files = [lidar_path + tk + ".bin" for tk in tokens]
#         anno_files = [annos_path + tk + ".csv" for tk in tokens]

#         lidar_generator = LidarDataLoader(lidar_files, ndim=5)
#         annos_generator = Box3dDataLoader(anno_files, filt_thres=0.1, is_track=True, is_waymo=True)
#         rp = RvizPublisher("sg_node", 10, lidar_generator, None, annos_generator, None, 0)

#         rp.Process(startfrom = 0)
        
    else:
        use_bin = 0
        lidar_path = "/home/zhanghao/code/GitLab/lidar_ped_veh_pipeline/test_seq_0/bin4/" if use_bin else "/mnt/data/waymo_opensets/val/lidar/"
        # image_path = "/mnt/data/waymo_opensets/val/image_res/"
        poses_path = "/mnt/data/waymo_opensets/val/annos/"
        
        annos_path = "/home/zhanghao/code/GL/trackingwithvelo/py/data/output_new/seq_all/giou-0.3/txt/"
        
        tokens = []
        # select_seq = [0]
        # select_seq = [161,163,164]
        select_seq = [i for i in range(0,202)]
        # for ii in range(6, 30):
        for ii in select_seq:
            seq_ii_num = len(glob.glob(lidar_path + "/seq_%d_frame_*"%ii))
            tokens += ["seq_%d_frame_"%ii + str(jj) for jj in range(seq_ii_num)]

        lidar_files = [lidar_path + tk + (".bin" if use_bin else ".pkl") for tk in tokens]
        pose_files = [poses_path + tk + ".pkl" for tk in tokens]
        anno_files = [annos_path + tk + ".txt" for tk in tokens]
        # anno_files = [annos_path + tk + ".pkl" for tk in tokens]
        # image_files = [image_path + tk + ".jpg" for tk in tokens]

        lidar_generator = LidarDataLoader(lidar_files, 4 if use_bin else 3)
        annos_generator = Box3dDataLoader(anno_files, filt_thres=0.1, is_track=True)
        # annos_generator = Box3dDataLoader(anno_files, filt_thres=0.1, is_track=False, is_waymo=True)
        # image_generator = ImageDataLoader(image_files)
        poses_generator = PoseLoader(pose_files)
        # rp = RvizPublisher("sg_node", 10, lidar_generator, image_generator, annos_generator, poses_generator, 0)
        rp = RvizPublisher("sg_node", 10, lidar_generator, None, annos_generator, poses_generator, 0)

        rp.Process(startfrom = 0)


        
###### ??????????????? pcd        
#     lidar_path = "/home/zhanghao/code/others/competition12/pcd/*"
#     lidar_files = sorted(glob.glob(lidar_path))
#     lidar_generator = LidarDataLoader(lidar_files, ndim=4)
#     rp = RvizPublisher("sg_node", 10, lidar_generator, None, None, field_angle=60)
#     rp.Process()