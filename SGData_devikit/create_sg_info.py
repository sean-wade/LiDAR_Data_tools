"""
    功能: 
        针对多文件夹组织的 SGData
        为每个文件夹生成专属的 info.pkl
            (如 train_001_20220806_infos.pkl 和 val_001_20220806_infos.pkl)
        最后合并成总的 info.pkl
            (如 train_total_infos.pkl 和 val_total_infos.pkl)

    使用:
        python tools/create_sg_infos.py \
            --sg_data_dir /mnt/data/SGTrain \
            --save_path /mnt/data/SGTrain/infos \
            --exclude 001_20220801 \
            --exclude 002_20220809
    
    参数说明: 
        --sg_data_dir, SG5 的数据集路径, 其下面是类似于 001_20220806 的文件夹, 该文件夹中至少包含(lidar_undist & labelled_lidar) 两个文件夹
        --save_path, infos 和 日志 保存的目录, 由于训练平台的数据盘没有写入权限, 请保存至其他位置
        --exclude, 生成时排除的子文件夹, (如 001_20220801 文件夹中的 json 文件标注格式不对，且与 001_20220806 重复, 因此需要排除)
"""
import os
import tqdm
import json
import glob
import torch
import shutil
import random
import logging
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
import matplotlib.pyplot as plt


# 确保每次随机 sample 的序列相同
random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description="ceate sg data info")
    parser.add_argument("--sg_data_dir", type=str, default="/home/jovyan/vol-2", help="path to sg data directory")
    parser.add_argument("--save_path", type=str, default="/home/jovyan/infos", help="path to the infos saving directory" )
    parser.add_argument("--subset", action='append', default=["004_20220818", "005_20220818"], help="sub name to exclude")
    args = parser.parse_args()
    return args


def obj2box(obj):
    box = [
        obj['bbox']['center_x'], 
        obj['bbox']['center_y'],
        obj['bbox']['center_z'],
        obj['bbox']['length'],
        obj['bbox']['width'],
        obj['bbox']['height'],
        0.0,                        # global speed_x
        0.0,                        # global speed_y
        obj['bbox']['heading']
    ]
    return box


def check_paths_time_consistency(data_paths, anno_paths):
    if len(data_paths) != len(anno_paths):
        logging.error("Error, paths length aren't the same %d, %d ! " % (len(data_paths), len(anno_paths) ) )
        return False
    for dp,ap in zip(data_paths, anno_paths):
        name1 =  os.path.basename(dp)
        name2 = os.path.basename(ap)
        if ".".join(name1.split('.')[:2]) != ".".join(name2.split('.')[:2]):
                logging.error("Error, different timestamp between %s and %s" %(name1, name2))
                return False
    return True


def generate_infos(ps, anno_ps):
    # root_path = os.path.join(root_path,"*bin") if not root_path.endswith("bin") else root_path
    # anno_path = os.path.join(anno_path,"*json") if not anno_path.endswith("json") else anno_path
    # ps = glob.glob(root_path)
    # anno_ps = glob.glob(anno_path)
    ps.sort()
    anno_ps.sort()
    logging.info("  %d data paths totally " % len(ps))
    logging.info("  checking consistency of  data paths and annotation paths . . .")
    check_paths_time_consistency(ps, anno_ps)
    ret = []
    logging.info('  generating infos . . .')
    for idx, (path,anno_path) in tqdm(enumerate(zip(ps, anno_ps))):
        data = json.load(open(anno_path,'r'))
        token =  os.path.basename(path)
        token_anno = os.path.basename(anno_path)
        assert ".".join(token.split('.')[:2]) == ".".join(token_anno.split('.')[:2]), "data path and anno path should have the exactly the same timestamp"
        
        timestamp = float(".".join(token.split('.')[:2]))
        boxes = []
        names = []
        ignores = []
        point_nums = []
        str2bool={'false':False,'true':True}
        for obj in data['objects']:
            # point_nums.append(obj['points']['lidar'])
            point_nums.append(obj['points'])                # 最新的标注 json 文件中这样访问
            boxes.append(obj2box(obj))
            names.append(obj['class'])
            #####################################################
            # 本期标注有些没有ignore, 暂时使用try
            try:
                ignores.append(str2bool[obj['attributes']['ignore']])
            except :
                ignores = [False] * len(names)
            #####################################################
        sweeps = []
        gt_boxes = np.array(boxes)
        gt_names = np.array(names)
        gt_ignores = np.array(ignores)
        gt_point_nums = np.array(point_nums).astype(np.int16)
        dic = {'path' : path, 'anno_path' : anno_path, 'token': token, 'timestamp' : timestamp, 'sweeps' : sweeps,
               'gt_boxes' : gt_boxes, 'gt_names':gt_names, 'gt_ignores':gt_ignores,
              'gt_point_nums' : gt_point_nums
              }
        ret.append(dic)
    return ret



def get_all_package_ids(label_dir):
    """
        获取某个 label_dir 下的所有 package_ids
    """
    package_ids = []
    labels = os.listdir(label_dir)
    for label_name in labels:
        package_ids.append(json.load(open(label_dir+"/"+label_name,'r'))["infos"]["lidar_package_id"])
    return list(set(package_ids))


def random_split_ids(id_list, ratio=0.8):
    id_nums = len(id_list)
    id_nums_train = int(ratio * id_nums)
    logging.info(f"    Total package nums = {id_nums}, train nums = {id_nums_train}")
    train_ids = random.sample(id_list, id_nums_train)
    val_ids = [iiid for iiid in id_list if iiid not in train_ids]
    return train_ids, val_ids


def get_files_by_pkgids(pkg_ids, subdir):
    label_dir = os.path.join(subdir, "labelled_lidar")
    lidar_dir = os.path.join(subdir, "lidar_undist")
    label_paths, lidar_paths = [], []
    label_names = os.listdir(label_dir)
    for label_name in label_names:
        label_full_path = os.path.join(label_dir, label_name)
        lidar_full_path = os.path.join(lidar_dir, label_name.replace(".json", ".bin"))
        cur_pkg_id = json.load(open(label_dir + "/" + label_name,'r'))["infos"]["lidar_package_id"]
        if cur_pkg_id in pkg_ids:
            label_paths.append(label_full_path)
            lidar_paths.append(lidar_full_path)
    return label_paths, lidar_paths


def main(args):
    valid_subdirs = [os.path.join(args.sg_data_dir, x) for x in args.subset]

    logging.info("Generating train and val infos by create_sg_infos.py.")
    logging.info("SG data dir : %s\n"%args.sg_data_dir)
    
    for subdir in valid_subdirs:
        curr_package_ids = get_all_package_ids(subdir + "/labelled_lidar")
        logging.info(f"Process {subdir}")
        train_ids, val_ids = random_split_ids(curr_package_ids)
        logging.info("    Train package ids : %s" % str(train_ids))
        logging.info("    Val package ids : %s" % str(val_ids))
        train_label_paths, train_lidar_paths = get_files_by_pkgids(train_ids, subdir)
        val_label_paths, val_lidar_paths = get_files_by_pkgids(val_ids, subdir)
        curr_info_train = generate_infos(train_lidar_paths, train_label_paths)
        curr_info_val = generate_infos(val_lidar_paths, val_label_paths)
        sub_name = subdir.split("/")[-1]
        pkl.dump(curr_info_train, open(args.save_path + f"/train_{sub_name}_infos.pkl",'wb'))
        pkl.dump(curr_info_val, open(args.save_path + f"/val_{sub_name}_infos.pkl",'wb'))
        logging.info("Finished %s ... \n\n\n" % subdir)

def set_logger():
    logging.basicConfig(level    = logging.INFO,                                                     
                        format   = '%(asctime)s  %(filename)s : %(levelname)s  %(message)s', 
                        datefmt  = '%Y-%m-%d %H:%M:%S',
                        filename = args.save_path + "/generate_infos.log",
                        filemode = 'w') 
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s  %(filename)s : %(levelname)s  %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    set_logger()
    main(args)
