import os
import sys
import pcl
import json
import pickle
import random
import operator
import numpy as np
from tqdm import tqdm
import os.path as osp
from copy import deepcopy
from utils import plot_dict, plot_list


class SGLidarDataset:
    NumPointFeatures = 4  # x, y, z, intensity

    def __init__(self, data_path="data/lidar", lidar_suffix=".bin"):
        self.data_path = data_path
        self.lidar_suffix = lidar_suffix

        print("data path : ", data_path)

        self.lidar_path = osp.join(data_path, "lidar_undist")
        self.label_path = osp.join(data_path, "labelled_lidar")
        assert osp.exists(self.lidar_path), f"{self.lidar_path} lidars doesnot exist !"
        assert osp.exists(self.label_path), f"{self.label_path} labels doesnot exist !"

        self.lidar_names = os.listdir(self.lidar_path)
        self.lidar_names.sort()
        self.lidar_nums = len(self.lidar_names)
        assert self.lidar_nums == len(os.listdir(self.label_path)), "label nums is not equal to lidar nums !"

        self.label_names = [ln.replace(lidar_suffix, ".json") for ln in self.lidar_names]
        self.idx = 0


    def check_continuance(self):
        # TODO: 检查数据连续性
        pass


    def load_label(self, index):
        label_path = osp.join(self.label_path, self.label_names[index])
        labels = json.load(open(label_path))

        return labels


    def load_lidar(self, index):
        lidar_path = osp.join(self.lidar_path, self.lidar_names[index])

        if self.lidar_suffix == ".bin":
            points = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, SGLidarDataset.NumPointFeatures))
            return points

        elif self.lidar_suffix == ".pcd":
            load_func = pcl.load if SGLidarDataset.NumPointFeatures==3 else pcl.load_XYZI
            points = load_func(lidar_path).to_array()
            return points


    def __len__(self):
        return self.lidar_nums


    def __iter__(self):
        return self


    def __next__(self):
        if self.idx == self.lidar_nums-1:
            raise StopIteration

        points = self.load_lidar(self.idx)
        labels = self.load_label(self.idx)

        self.idx += 1
        return points, labels, self.lidar_names[self.idx]


    def visualize_in_rviz(self, ros_rate=10, field_angle=60, start_from=0):
        import rospy 
        import publish_utils
        from frame_data import FrameBoxData

        rospy.init_node("sg_lidar_dataset", anonymous=True)
        pcl_pub = rospy.Publisher('my_point_cloud', publish_utils.PointCloud2, queue_size=10)
        box_pub = rospy.Publisher('my_box3d', publish_utils.MarkerArray, queue_size=1)
        ego_pub = rospy.Publisher('my_ego_view', publish_utils.Marker, queue_size=10)
        rate = rospy.Rate(ros_rate)

        self.idx = start_from
        while not rospy.is_shutdown():
            pts, lbs, f_name = self.__next__()
            print(self.idx, f_name, pts.shape, len(lbs["objects"]))

            # publish 3d-bbox
            box_infos = FrameBoxData(is_waymo=True)
            box_infos.from_sg_json(lbs)
            publish_utils.publish_3dbox(box_pub, box_infos, pose=None, Lifetime=1.0/ros_rate)

            # publish field angle
            if field_angle > 0:
                publish_utils.publish_ego_car(ego_pub, field_angle)

            # publish point cloud
            publish_utils.publish_point_cloud(pcl_pub, pts, down_ratio=5)
            rate.sleep()


    def visualize_in_bev_rgb(self, bev_img_save_dir="./bev_img"):
        import cv2
        from utils import point_cloud_2_birdseye, plot_sgobjs_on_image

        # save_dir = osp.join(self.data_path, "bev_rgb")
        # os.makedirs(save_dir, exist_ok=True)
        for pts, lbs, f_name in tqdm(self):
            output_filename = f_name.replace(".bin", ".png")
            output_filepath = osp.join(bev_img_save_dir, output_filename)

            image_bev = point_cloud_2_birdseye(pts)

            image_bev_objs = plot_sgobjs_on_image(image_bev, lbs)

            cv2.imwrite(output_filepath, image_bev_objs)


    def statistics(self, img_save_dir="./statistics"):
        """
            Function: 
                1. objs' class distribution
                2. objs' points-num distribution
                3. objs' occlusion distribution
                3. objs' ignore distribution
        """
        class_count = {}
        ignore_count = {}
        occlusion_count = {}
        points_num_count = []

        for _, lbs, f_name in self:
            for obj in lbs["objects"]:
                points_num_count.append(obj["points"]["lidar"])

                if obj["class"] not in class_count:
                    class_count[obj["class"]] = 1
                else:
                    class_count[obj["class"]] += 1

                is_ignore = obj["attributes"].get("ignore", "false")
                if is_ignore not in ignore_count:
                    ignore_count[is_ignore] = 1
                else:
                    ignore_count[is_ignore] += 1

                occ = obj["attributes"]["occlusion"]
                if occ not in occlusion_count:
                    occlusion_count[occ] = 1
                else:
                    occlusion_count[occ] += 1

        stat_path = img_save_dir  # osp.join(self.data_path, "statistics")
        os.makedirs(stat_path, exist_ok=True)

        with open(osp.join(stat_path, "stat.txt"), "w") as fff:
            fff.writelines(str(class_count) + "\n")
            fff.writelines("ignore : " + str(ignore_count) + "\n")
            fff.writelines(str(occlusion_count) + "\n")
            fff.writelines("Total objects : " + str(len(points_num_count)) + "\n")
        
        print(open(osp.join(stat_path, "stat.txt"), "r").read())

        plot_dict(class_count, "classes", osp.join(stat_path, "classes_count.png"))
        plot_dict(ignore_count, "ignore", osp.join(stat_path, "ignore_count.png"))
        plot_dict(occlusion_count, "occlusion", osp.join(stat_path, "occlusion_count.png"))
        plot_list(points_num_count, "points_num_count", osp.join(stat_path, "points_num_count.png"))


    def create_train_val_split(self, val_ratio=0.2, seq_length=20, txt_save_dir="./split"):
        """
        Function:
            Create 80%-train and 20%-val split.
            The val-set are pick by continuance 20 frames per fraction.
        """
        import random
        train_names = []
        val_names = []

        seq_num = int(self.lidar_nums / seq_length)
        val_seq_num = int(val_ratio * seq_num)

        val_select_seq = random.sample(list(range(seq_num)), val_seq_num)

        for i in range(seq_num):
            if i in val_select_seq:
                val_names += self.lidar_names[20*i:(i+1)*20]
            else:
                train_names += self.lidar_names[20*i:(i+1)*20]
        
        train_names += self.lidar_names[seq_num*20:]

        with open(txt_save_dir + "/val.txt", "w") as val_txt:
            for val_name in val_names:
                val_txt.write(val_name[:-4] + "\n")

        with open(txt_save_dir + "/train.txt", "w") as val_txt:
            for train_name in train_names:
                val_txt.write(train_name[:-4] + "\n")

        print("Success split train[size=%d] & val[size=%d] set. Txts are saved under data_path."%(len(train_names), len(val_names)))


    def convert_to_det3d(self):
        """
        Convert to det3d format(all data are pickle files.). Such as follow:
            ├── train
                ├── annos
                ├── lidar
            └── val
                ├── annos
                └── lidar
        """
        pass


if __name__ == "__main__":
    sgld = SGLidarDataset("./data/lidar")

    sgld.visualize_in_rviz(ros_rate=1, start_from=2700)

    # sgld.statistics("./stat_img")

    # sgld.create_train_val_split("./split_txts")

    # sgld.visualize_in_bev_rgb("./bev_img")
