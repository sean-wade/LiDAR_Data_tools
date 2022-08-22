#!/usr/bin/python2
# -*- coding:utf-8 -*-
import json
import pickle
import numpy as np


WAYMO_GT_LABEL_DICT = {0:"unk", 1:"car", 2:"ped", 3:"sig", 4:"cyc"}
WAYMO_DT_LABEL_DICT = {0:'car', 1:'ped', 2:'cyc', 3:"tra", 4:"other"}

for jj in range(5,100):
    WAYMO_GT_LABEL_DICT.update({jj:str(jj)})
    WAYMO_DT_LABEL_DICT.update({jj:str(jj)})

WAYMO_DT_LABEL_DICT_INV = {
    "CAR" : 0,
    "PEDESTRIAN" : 1,
    "CYCLIST" : 2
}

SG_DET_LABEL_NAMES = ["Car","Truck","Bus","Construction","Trailer","Tricycle","Cyclist","Pedestrian","Traffic_cone"]
SG_DET_LABEL_DICT = {
    i : SG_DET_LABEL_NAMES[i] for i in range(len(SG_DET_LABEL_NAMES))
}
class FrameBoxData(object):
    def __init__(self, is_waymo=True):
        self.boxes3d = np.empty(0)
        self.scores = np.empty(0)
        self.label_ids = np.empty(0)
        self.track_ids = []
        self.label_names = []
        self.global_velos = np.empty(0)
        self.local_velos = np.empty(0)
        self.is_waymo = is_waymo
        self.points_nums = None


    def filter_by_score(self, score_thres):
        mask = self.scores >= score_thres
        self.scores = self.scores[mask]
        self.boxes3d = self.boxes3d[mask]
        self.label_ids = self.label_ids[mask]
        if len(self.track_ids) > 0:
            self.track_ids = [self.track_ids[mm] for mm in range(len(mask)) if mask[mm]]
        
        if len(self.label_names) > 0:
            self.label_names = [self.label_names[mm] for mm in range(len(mask)) if mask[mm]]

        if len(self.global_velos) > 0:
            self.global_velos = self.global_velos[mask]
        if len(self.local_velos) > 0:
            self.local_velos = self.local_velos[mask]
        
        
    def filter_by_points_nums(self, min_num=5):
        if self.points_nums is None:
            return 
        
        mask = self.points_nums >= min_num
        self.scores = self.scores[mask]
        self.boxes3d = self.boxes3d[mask]
        self.label_ids = self.label_ids[mask]
        if len(self.track_ids) > 0:
            self.track_ids = [self.track_ids[mm] for mm in range(len(mask)) if mask[mm]]
        
        if len(self.label_names) > 0:
            self.label_names = [self.label_names[mm] for mm in range(len(mask)) if mask[mm]]

        if len(self.global_velos) > 0:
            self.global_velos = self.global_velos[mask]
        if len(self.local_velos) > 0:
            self.local_velos = self.local_velos[mask]
        
        
        
    def load_from_file(self, file_path, track=False):
        """
        Args:
            file_path (str): pkl or txt(numpy)
        """
        print("annos path: ", file_path)
        if file_path.endswith(".txt"):
            if track:
                self.load_waymo_tk_txt(file_path)
            else:
                self.load_det3d_pd_txt(file_path)
        elif file_path.endswith(".csv"):
            self.load_sg_api_csv(file_path)
        elif file_path.endswith(".pkl"):
            self.load_waymo_dets_pkl(file_path, "box3d_lidar")
        elif file_path.endswith(".bin"):
            self.load_waymo_dets_pkl(file_path, "box3d_lidar", SG_DET_LABEL_DICT)
        elif file_path.endswith(".json"):
            json_data = json.load(open(file_path))
            self.from_sg_json(json_data)

            
    def load_waymo_gt_txt(self, txt_path):
        """
        Args:
            txt_path (str): colomns=[id, label, x,y,z,dx,dy,dz,theta, 1, global_vx, global_vy]
        """
        npy_datas = np.loadtxt(txt_path).reshape(-1, 12)
        self.boxes3d = npy_datas[:, 2:9]
        self.label_ids = npy_datas[:, 1]
        self.track_ids = [str(ii) for ii in npy_datas[:, 0]]
        self.label_names = [WAYMO_GT_LABEL_DICT[int(ii)] for ii in self.label_ids]
        self.scores = npy_datas[:, 9]
        self.global_velos = npy_datas[:, 10:12]
        
        if npy_datas.shape[1] > 12:
            self.points_nums = npy_datas[:, 12]


    def load_sg_api_csv(self, csv_path):
        """
        Args:
            csv_path(str): colomns=[uuid, age, time, type, label, x,y,z,dx,dy,dz,theta, score, global_vx, global_vy]
        """
        npy_datas = np.loadtxt(csv_path, delimiter=',', usecols=[5,6,7,8,9,10,11,12,13,14]).reshape(-1, 10)
        self.boxes3d = npy_datas[:, 0:7]
        self.label_names = list(np.loadtxt(csv_path, delimiter=',', usecols=[4], dtype=np.str).reshape(-1))
        self.label_ids = np.array([WAYMO_DT_LABEL_DICT_INV[ii] for ii in self.label_names])
        self.scores = npy_datas[:, -3]
        self.global_velos = npy_datas[:, -2:]
        idd = np.loadtxt(csv_path, delimiter=',', usecols=[0], dtype=np.str).reshape(-1)
        self.track_ids = [iii for iii in idd]
        # self.track_ids = [hash(iii)%999 for iii in idd]
        


    def load_waymo_tk_txt(self, txt_path):
        """
        Args:
            txt_path (str): colomns=[id, label, x,y,z,dx,dy,dz,theta, score, global_vx, global_vy]
        """
        self.load_waymo_gt_txt(txt_path)
        self.label_names = [WAYMO_DT_LABEL_DICT[int(ii)] for ii in self.label_ids]
        # npy_datas = np.loadtxt(txt_path).reshape(-1, 12)
        # self.boxes3d = npy_datas[:, 2:9]
        # self.label_ids = npy_datas[:, 1]
        # self.track_ids = [str(ii) for ii in npy_datas[:, 0]]
        # self.label_names = [WAYMO_GT_LABEL_DICT[int(ii)] for ii in self.label_ids]
        # self.scores = npy_datas[:, 9]
        # self.global_velos = npy_datas[:, 10:]
    

    def load_det3d_pd_txt(self, txt_path):
        npy_datas = np.loadtxt(txt_path).reshape(-1, 9)
        self.boxes3d = npy_datas[:, :7]
        self.scores = npy_datas[:, 7]
        self.label_ids = npy_datas[:, 8]
        self.label_names = [WAYMO_DT_LABEL_DICT[int(ii)] for ii in self.label_ids]        


    def load_waymo_dets_pkl(self, pkl_path, key="box3d_lidar", class_name_dict=WAYMO_DT_LABEL_DICT):
        """[only work in python3]
        Args:
            pkl_path (str): [pickle file(with objs detected in one frame), generated from det3d]
        """
        data = pickle.load(open(pkl_path, 'rb'))
        
        if key != "box3d_lidar":
            boxes3d = np.concatenate([obj['box'] for obj in data['objects']] ,axis=0).reshape(-1,9)
            self.boxes3d = boxes3d[:, [0,1,2,3,4,5,-1]]
            self.track_ids = [hash(obj['name'])%1999 for obj in data['objects']]
            labels = [obj['label'] for obj in data['objects']]
            self.label_names = [class_name_dict[i] for i in labels]
            self.label_ids = np.array(labels)
            # self.global_velos = boxes3d[:, [6,7]]
        else:
            self.boxes3d = data['box3d_lidar'][:, [0,1,2,3,4,5,-1]]
            self.boxes3d[:, -1] *= -1    # to tackle a bug on sg-train data. @2022.08.09
            labels = data['label_preds']
            self.label_names = [class_name_dict.get(i, 4) for i in labels]
            self.scores = data['scores']
            self.label_ids = np.array(labels)
            if data['box3d_lidar'].shape[1] > 7:
                self.local_velos = data['box3d_lidar'][:, [6,7]]
            else:
                self.local_velos = np.zeros((data['box3d_lidar'].shape[0], 2))
            
            # print(self.boxes3d, self.scores)
    
    
    def from_pandaset(self, panda_anno):
        boxes = []
        labels = []
        scores = []
        velos = []
        for idx, dd in panda_anno.iterrows():
            if dd['cuboids.sensor_id'] in [0,1,-1]:
                boxes.append([dd['position.x'],
                              dd['position.y'],
                              dd['position.z'],
                              dd['dimensions.y'],
                              dd['dimensions.x'],
                              dd['dimensions.z'],
                              dd['yaw'] + np.pi/2.0
                             ])
                self.label_names.append(dd['label'])
                labels.append(0)
                self.track_ids.append(dd['uuid'])
                scores.append(1)
                velos.append([0,0])
                
        self.boxes3d = np.array(boxes)
        self.label_ids = np.array(labels)
        self.scores = np.array(scores)
        self.global_velos = np.array(velos)

        
    def from_sg_json(self, json_data):
        if isinstance(json_data, str):
            json_data = json.load(open(json_data))

        if len(json_data["objects"]) == 0:
            return

        boxes = []
        labels = []
        scores = []
        velos = []
        for obj in json_data["objects"]:
            self.label_names.append(obj["class"])
            self.track_ids.append(obj['track_id'])
            scores.append(1)
            labels.append(0)
            boxes.append([
                            obj['bbox']['center_x'],
                            obj['bbox']['center_y'],
                            obj['bbox']['center_z'],
                            obj['bbox']['length'],
                            obj['bbox']['width'],
                            obj['bbox']['height'],
                            obj['bbox']['heading'],
                         ])

        self.boxes3d = np.array(boxes)
        self.label_ids = np.array(labels)
        self.scores = np.array(scores)
        
        
class Box3dDataLoader(object):
    def __init__(self, info_files, filt_thres=0.5, is_track=False, is_waymo=True):
        """[summary]

        Args:
            info_files (list): [list of full_path with .pkl/.txt extension files]
            # ! < MUST SORT > !
        """
        self.idx = 0
        self.filt_thres = filt_thres
        self.info_files = info_files
        self.is_track   = is_track
        self.is_waymo   = is_waymo
        self.info_nums  = len(self.info_files)
        print("Box3dDataLoader Init with [%d] files"%self.info_nums)

    
    def __len__(self):
        return self.info_nums


    def __iter__(self):
        return self

    
    def __next__(self):
        """
        Returns:
            FrameBoxData
        """
        if self.idx == self.info_nums-1:
            raise StopIteration

        info_path = self.info_files[self.idx]
        frame_box_infos = FrameBoxData(self.is_waymo)
        # if self.is_track:
        #     frame_box_infos.load_waymo_tk_txt(info_path)
        # else:
        #     frame_box_infos.load_det3d_pd_txt(info_path)
        print(info_path)
        
        frame_box_infos.load_from_file(info_path, self.is_track)
        frame_box_infos.filter_by_score(self.filt_thres)

        self.idx += 1
        return frame_box_infos


    def load_idx(self, idx):
        if idx >= self.info_nums-1:
            print("Load info index[%d] out of range[%d] !!!"%(idx, self.info_nums))
            return None

        info_path = self.info_files[idx]
        frame_box_infos = FrameBoxData(self.is_waymo)
        # if self.is_track:
        #     frame_box_infos.load_waymo_tk_txt(info_path)
        # else:
        #     frame_box_infos.load_det3d_pd_txt(info_path)
        
        frame_box_infos.load_from_file(info_path, self.is_track)
        frame_box_infos.filter_by_score(self.filt_thres)
        return frame_box_infos


if __name__ == "__main__":
    # fds = FrameBoxData()
    # fds.load_waymo_gt_txt("/mnt/data/waymo_opensets/val/annos_txt/seq_0_frame_0.txt")
    # print(fds.track_ids)


    # fds2 = FrameBoxData()
    # fds2.load_waymo_tk_txt("/mnt/data/waymo_opensets/val/for_cpp/result/result_py_2s/seq_0_frame_0.txt")
    # print(fds2.track_ids)


    import os
    import glob

    files = []
    for ii in range(202):
        files += sorted(glob.glob("/mnt/data/waymo_opensets/val/annos_txt/seq_%d_frame_*")%ii, key=lambda x: int(x.split("/")[-1][:-4].split("_")[-1]))

    # print(files[:20])

    box_loader = Box3dDataLoader(files, is_track=True)
    iii = 0
    while True:
        rrr = box_loader.load_idx(iii)
        print(rrr.track_ids)
        if iii == 2:
            break
        iii += 1




