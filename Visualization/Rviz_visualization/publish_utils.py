#!/usr/bin/python3
# -*- coding:utf-8 -*-

# import tf
import cv2
import sys
import numpy as np

import rospy 
from cv_bridge import CvBridge
from std_msgs.msg import Header

from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, PointCloud2, Imu, NavSatFix, PointField

from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pcl2

from box_utils import compute_3d_corners_kitti, convert_kitti_waymo, vector_rotate


FRAME_ID = "lidar" # the base coordinate name in rviz
DETECTION_COLOR_MAP = {0: (1,1,0), 
                       1: (0, 1, 1), 
                       2: (0.5, 0.2, 1),
                       3:(0,1,0),
                       4:(0.2,1,0.2),
                       5:(0.5,0.2,0.2),
                       6:(0.2,0.5,0.2),
                       7:(0.4,1,0.4),
                       8:(0.8,1,0.8),
                      } # color for detection, in format bgr

# TRACKING_COLOR_MAP
NUM_TRACK_COLORS = 256
colors = np.random.uniform(0,1,size=(NUM_TRACK_COLORS , 3))
TRACKING_COLOR_MAP = {i:colors[i].tolist() for i in range(NUM_TRACK_COLORS)}


# connect vertic
LINES = [[0, 1], [1, 2], [2, 3], [3, 0]] # lower face
LINES+= [[4, 5], [5, 6], [6, 7], [7, 4]] # upper face
LINES+= [[4, 0], [5, 1], [6, 2], [7, 3]] # connect lower face and upper face
LINES+= [[1, 6], [2, 5]] # front face and draw x


def publish_camera(cam_pub, bridge, image, borders_2d_cam2s=None, object_types=None, log=False):
    """
    Publish image in bgr8 format
    If borders_2d_cam2s is not None, publish also 2d boxes with color specified by object_types
    If object_types is None, set all color to cyan
    """
    if borders_2d_cam2s is not None:
        for i, box in enumerate(borders_2d_cam2s):
            top_left = int(box[0]), int(box[1])
            bottom_right = int(box[2]), int(box[3])
            if object_types is None:
                cv2.rectangle(image, top_left, bottom_right, (255,255,0), 2)
            else:
                cv2.rectangle(image, top_left, bottom_right, DETECTION_COLOR_MAP[object_types[i]], 2) 
    cam_pub.publish(bridge.cv2_to_imgmsg(image, "bgr8"))


def publish_point_cloud2(pcl_pub,point_cloud):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = FRAME_ID
    pcl_pub.publish(pcl2.create_cloud_xyz32(header, point_cloud[:,:3]))


def publish_point_cloud(pcl_pub, point_cloud, down_ratio=3):
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = FRAME_ID

    if point_cloud.shape[-1] > 3:
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1)
            ]
        pcl_pub.publish(pcl2.create_cloud(header, fields, point_cloud[::down_ratio, :4]))    # [::3] 下采样3倍
    else:
        pcl_pub.publish(pcl2.create_cloud_xyz32(header, point_cloud[::down_ratio, :4]))      # [::3] 下采样3倍


def publish_ego_car(ego_car_pub, field_angle=90):
    # publish left and right 45 field_angle FOV lines and ego car model mesh
    marker = Marker()
    marker.header.frame_id = FRAME_ID
    marker.header.stamp = rospy.Time.now()

    marker.id = 0
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration()
    marker.type = Marker.LINE_STRIP
    # line
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    marker.scale.x = 0.2 # line width
    marker.points = []
    # check the kitti axis model 
    
    degree = field_angle / 2.0 / 180.0 * np.pi
    # marker.points.append(Point(20,-20*np.tan(degree), 0)) # left up
    marker.points.append(Point(20 * np.cos(degree), -20*np.sin(degree), 0)) # left up
    marker.points.append(Point(0,0,0)) # center
    # marker.points.append(Point(20, 20*np.tan(degree), 0)) # right up
    marker.points.append(Point(20 * np.cos(degree), 20*np.sin(degree), 0)) # right up
    ego_car_pub.publish(marker)


# def publish_imu(imu_pub, imu_data, log=False):
#     """
#     Publish IMU data
#     http://docs.ros.org/melodic/api/sensor_msgs/html/msg/Imu.html
#     """
#     imu = Imu()
#     imu.header.frame_id = FRAME_ID
#     imu.header.stamp = rospy.Time.now()
#     q = tf.transformations.quaternion_from_euler(float(imu_data.roll), float(imu_data.pitch), \
#                                                      float(imu_data.yaw)) # prevent the data from being overwritten
#     imu.orientation.x = q[0]
#     imu.orientation.y = q[1]
#     imu.orientation.z = q[2]
#     imu.orientation.w = q[3]
#     imu.linear_acceleration.x = imu_data.af
#     imu.linear_acceleration.y = imu_data.al
#     imu.linear_acceleration.z = imu_data.au
#     imu.angular_velocity.x = imu_data.wf
#     imu.angular_velocity.y = imu_data.wl
#     imu.angular_velocity.z = imu_data.wu

#     imu_pub.publish(imu)
#     if log:
#         rospy.loginfo("imu msg published")


def publish_gps(gps_pub, gps_data, log=False):
    """
    Publish GPS data
    """
    gps = NavSatFix()
    gps.header.frame_id = FRAME_ID
    gps.header.stamp = rospy.Time.now()
    gps.latitude = gps_data.lat
    gps.longitude = gps_data.lon
    gps.altitude = gps_data.alt

    gps_pub.publish(gps)
    if log:
        rospy.loginfo("gps msg published")


# def publish_3dbox(box3d_pub, corners_3d_velos, texts=None, types=None, speeds=None, track_color=False, Lifetime=100):
def publish_3dbox(box3d_pub, frame_box_infos, pose=None, Lifetime=100):
    if len(frame_box_infos.boxes3d) == 0:
        return
    
    color = (0.6, 0.2, 0.6)
    if frame_box_infos.is_waymo:
        boxes_local = convert_kitti_waymo(frame_box_infos.boxes3d)
    else:
        boxes_local = frame_box_infos.boxes3d

    corners_3d_velos = compute_3d_corners_kitti(boxes_local)
    if frame_box_infos.global_velos.any() and (pose is not None):
        # print(pose)
        angle = np.arctan2(-pose[1, 0], pose[0, 0])
        # angle = 0
        frame_box_infos.local_velos = vector_rotate(frame_box_infos.global_velos, angle)
    
    # 瞎调试用的
    # frame_box_infos.local_velos = frame_box_infos.global_velos
    # print(frame_box_infos.local_velos)

    marker_array = MarkerArray()
    for i, corners_3d_velo in enumerate(corners_3d_velos):
        # corners_3d_velo : 8 x 3， 8 corners
        text = frame_box_infos.label_names[i]   # + " " + str(frame_box_infos.scores[i])[:4]
        if len(frame_box_infos.track_ids) > 0:
            # Have track_id
            id2color = abs(hash(frame_box_infos.track_ids[i])) % NUM_TRACK_COLORS
            color = TRACKING_COLOR_MAP[id2color]

            # #####################
            # idff = float(frame_box_infos.track_ids[i])
            # if idff != 25:
            #     continue
            # #####################

            text = text + " " + str(frame_box_infos.track_ids[i])[:5].split(".")[0]
            # text = str(frame_box_infos.track_ids[i])[:3]

        elif len(frame_box_infos.label_ids)>0:
            t = int(frame_box_infos.label_ids[i])
            # print("label id = ", t)
            color = DETECTION_COLOR_MAP.get(t, (0.5,1,0.2))
            # text = frame_box_infos.label_names[i]

        marker = Marker()
        marker.header.frame_id = FRAME_ID
        marker.header.stamp = rospy.Time.now()
        marker.id = i
        marker.action = Marker.ADD
        marker.lifetime = rospy.Duration(Lifetime)  # 100ms
        # 将角点连成框线，主要使用marker.points 来收集角点信息
        marker.type = Marker.LINE_LIST
        marker.color.r = color[2]
        marker.color.g = color[1]
        marker.color.b = color[0]
        marker.color.a = 1.0
        marker.scale.x = 0.1

        marker.points = []
        # print(corners_3d_velo)
        for l in LINES:
            # print("corners_3d_velo = ", corners_3d_velo)
            p1 = corners_3d_velo[l[0]]
            marker.points.append(Point(p1[0], p1[1], p1[2]))
            p2 = corners_3d_velo[l[1]]
            # print(l[0], l[1])
            # print("- "*50)
            marker.points.append(Point(p2[0], p2[1], p2[2]))
        marker_array.markers.append(marker)

        # add score or other infos
        if text is not None:
            text_marker = get_text_marker(text, 
                                          Lifetime, 
                                          (corners_3d_velo[4][0], corners_3d_velo[4][1], corners_3d_velo[4][2]), 
                                          i+1000, 
                                          color)
            marker_array.markers.append(text_marker)

        if frame_box_infos.local_velos.any():
            # speed arrow
            p_start = ((corners_3d_velo[4] + corners_3d_velo[2]) / 2.0)[:2]
            p_end = p_start + frame_box_infos.local_velos[i] * 1
            Point_start = Point(p_start[0], p_start[1], 1)
            Point_end = Point(p_end[0], p_end[1], 1)
            speed_marker = get_arrow_marker(Point_start, Point_end, Lifetime, i+2000, (0,255,0))
            
            # speed norm text
            speed_norm = (frame_box_infos.local_velos[i][0] ** 2 + frame_box_infos.local_velos[i][1] ** 2) ** 0.5
            speed_norm = 0 if speed_norm < 0.02 else speed_norm * 3.6
            speed_text_marker = get_text_marker(str(speed_norm)[:4] + "km/h", 
                                                Lifetime, 
                                                (p_start[0], p_start[1], 2), 
                                                i+3000, 
                                                (0,1,1))

            marker_array.markers.append(speed_marker)                                    
            marker_array.markers.append(speed_text_marker)
        
    box3d_pub.publish(marker_array)


def publish_centers(center_pub, centers, radius=1, Lifetime=100, texts=None, types=None, track_color=False,):
    marker_array = MarkerArray()
    for i, center in enumerate(centers):
        # get color
        if types is None:
            color = (0.6, 0.2, 0.6)
        elif track_color:
            t = int(types[i] % NUM_TRACK_COLORS)
            color = TRACKING_COLOR_MAP[t]
        else:
            t = int(types[i])
            color = DETECTION_COLOR_MAP[t]/255.0

        # pub ball 
        ball_marker = get_ball_marker(radius, 
                                      Lifetime, 
                                      center, 
                                      i + 100, 
                                      color)
        marker_array.markers.append(ball_marker)

        # add score or other infos
        if texts is not None:
            text_marker = get_text_marker(texts[i], 
                                          Lifetime, 
                                          (center[0], center[1], center[2] + 0.5), 
                                          i+10000, 
                                          color)
            marker_array.markers.append(text_marker)


    center_pub.publish(marker_array)


def get_ball_marker(radius, Lifetime=1, position=(0,0,0), id=101, color=(255,0,0)):
    marker = Marker()
    marker.header.frame_id = FRAME_ID
    marker.header.stamp = rospy.Time.now()

    marker.id = id
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration(Lifetime)
    marker.type = Marker.SPHERE

    b, g, r = color
    marker.color.r = r
    marker.color.g = g
    marker.color.b = b
    marker.color.a = 1.0

    marker.scale.x = radius
    marker.scale.y = radius
    marker.scale.z = radius

    # 圆球中心在世界坐标系中的位置
    marker.pose.position.x = position[0]
    marker.pose.position.y = position[1]
    marker.pose.position.z = position[2]
    
    return marker


def get_text_marker(text="", Lifetime=1, position=(0,0,0), id=101, color=(255,0,0)):
    text_marker = Marker()
    text_marker.header.frame_id = FRAME_ID
    text_marker.header.stamp = rospy.Time.now()

    text_marker.id = id
    text_marker.action = Marker.ADD
    text_marker.lifetime = rospy.Duration(Lifetime)
    text_marker.type = Marker.TEXT_VIEW_FACING

    # 文字所在的位置
    text_marker.pose.position.x = position[0]
    text_marker.pose.position.y = position[1]
    text_marker.pose.position.z = position[2] + 0.5
    
    # 文字大小
    text_marker.scale.x = 2
    text_marker.scale.y = 2
    text_marker.scale.z = 2

    b, g, r = color
    text_marker.color.r = r
    text_marker.color.g = g
    text_marker.color.b = b
    text_marker.color.a = 1.0

    # 文字内容
    text_marker.text = str(text)
    return text_marker


def get_arrow_marker(p_start, p_end, Lifetime=1, id=1001, color=(255,0,0)):
    arrow_marker = Marker()
    arrow_marker.header.frame_id = FRAME_ID
    arrow_marker.header.stamp = rospy.Time.now()

    arrow_marker.id = id  # i + 2000
    arrow_marker.action = Marker.ADD
    arrow_marker.lifetime = rospy.Duration(Lifetime)
    arrow_marker.type = Marker.ARROW
    arrow_marker.points = []

    arrow_marker.color.r = color[0]
    arrow_marker.color.g = color[1]
    arrow_marker.color.b = color[2]
    arrow_marker.color.a = 1.0
    arrow_marker.scale.x = 0.3
    arrow_marker.scale.y = 0.8

    # p_start = ((corners_3d_velo[4] + corners_3d_velo[2]) / 2.0)[:2]
    # p_end = p_start + speeds[i] * 2
    # Point(p_start[0], p_start[1], 1)
    # Point(p_end[0], p_end[1], 1)

    arrow_marker.points.append(p_start)
    arrow_marker.points.append(p_end)
    return arrow_marker
