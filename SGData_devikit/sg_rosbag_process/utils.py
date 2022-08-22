import os
import cv2
import yaml
import numpy as np
from cv_bridge import CvBridgeError
from sensor_msgs.msg import Image
import sensor_msgs.point_cloud2 as pc2

try:
    import pcl
except Exception as e:
    print("If pcl library has a problem, plz save as numpy-bin file !")


try:
    import coloredlogs, logging
    logger = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG', logger=logger)
except:
    import logging as logger
    logger.basicConfig(level=logger.INFO)


def save_lidar_msg(msg, save_path, save_as_pcd=False):
    src_points = pc2.read_points_list(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)

    if save_as_pcd:
        pc = pcl.PointCloud_PointXYZI(src_points)
        pcl.save(pc, save_path)
    else:
        points = np.array(src_points, dtype = np.float32)
        points.tofile(save_path)


def save_image_msg(msg, bridge, save_path, intrinsics=None, distcoeffs=None):
    try:
        cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
    except:
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    
    if (intrinsics is not None) and (distcoeffs is not None):
        image_undistort = cv_image.copy()
        cv2.undistort(cv_image, intrinsics, distcoeffs, image_undistort, None)
        cv2.imwrite(save_path, image_undistort)
    else:
        cv2.imwrite(save_path, cv_image)


def save_ins_msg(msg, save_path):
    with open(save_path, "w") as txt:
        txt.write(str(msg))


def save_imu_msg(msg, save_path):
    with open(save_path, "w") as txt:
        txt.write(str(msg))


def check_folder_file_consistant(folder, hz):
    is_consistant = True
    assert os.path.exists(folder), "During check time consistant, cannot find %s"%folder
    f_names = os.listdir(folder)
    f_names.sort()
    logger.info("[%d] frames in sensor <%s>."%(len(f_names), folder))
    tss = [float(x[:-5]) for x in f_names]
    for i in range(len(tss) - 1):
        time_diff = tss[i+1] - tss[i]
        if not (hz * 0.8 < time_diff < hz * 1.2):
            logger.error("Unconsistant timestamp found, plz check <%s>:[%s=>%s]"%(folder, f_names[i], f_names[i+1]))
            is_consistant = False
    
    return is_consistant


def load_intrin_and_coef(yaml_path):
    if (yaml_path is None) or (not os.path.exists(yaml_path)):
        return None, None

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
        if data is None:
            return None, None
        intrinsics = data.get("IntrinsicsMatrix")
        distortion = data.get("Distortion")
        if intrinsics is None or distortion is None:
            return None, None
        else:
            return np.array(intrinsics).reshape(3,4)[:, :3], np.array(distortion)
        