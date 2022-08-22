#!/usr/bin/python3

import cv2
# import pcl
import numpy as np
import open3d as o3d
from lidar_camera_sync import LidarCameraSync


# SG5 calib cam-60. Details can ask SG5 calib group.
ex = np.array([[-0.0255,  -0.9997,    -0.0053,  -0.0279],
               [ 0.0057,   0.0051,    -1.0000,  -0.1244],
               [ 0.9997,  -0.0255,     0.0056,  -0.1017],
               [      0,        0,          0,        1],
              ])

ix = np.array([[1.9768e3,   1.1512, 990.2660],
               [       0, 1.9770e3, 549.7352],
               [       0,        0,        1]
               ])
# distort
k1, k2, k3, p1, p2 = -0.5574, 0.3210, -0.1407, 0.0021, -0.0031


if __name__ == "__main__":

    # image distort
    dist_coef = np.array([k1, k2, p1, p2, k3])
    image_ori = cv2.imread("./data/1642644263.200000.jpg")
    image_distort = image_ori.copy()
    cv2.undistort(image_ori, ix[:, :3], dist_coef, image_distort, None)

    # load pcd
    pcd = o3d.io.read_point_cloud("./data/1642644263.199739933.pcd")
    points = np.asarray(pcd.points)

    # load box
    boxes = np.loadtxt("./data/1642644263.199739933.txt")[:, :7]

    image_distort2 = image_distort.copy()
    image_distort3 = image_distort.copy()

    lcs = LidarCameraSync(ex, ix)
    image_proj, _, _ = lcs.proj_point_to_image(points, image_distort2)
    image_boxes = lcs.proj_box_to_image(boxes, image_distort3)

    cv2.imshow("original", image_ori)
    cv2.imshow("distort", image_distort)
    cv2.imshow("projection", image_proj)
    cv2.imshow("boxes", image_boxes)
    cv2.waitKey(0)





