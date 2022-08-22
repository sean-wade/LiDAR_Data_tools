import os
import pandaset
from publish_utils import *
from box3d_loader import FrameBoxData


class RvizPandasetPublisher(object):
    def __init__(self,
                 node_name   = "pandaset_node", 
                 data_path   = "/mnt/data/PandaSet/Data",
                 seq_idx     = 0,
                 lidar_idx   = 1,
                 ros_rate    = 10,
                 field_angle = 0,
                 to_ego = False,
                 ):
        self.ros_rate    = ros_rate
        self.field_angle = field_angle
        
        rospy.init_node(node_name, anonymous=True)
        # self.img_pub = rospy.Publisher('my_image', Image, queue_size=10)
        self.pcl_pub = rospy.Publisher('my_point_cloud', PointCloud2, queue_size=10)
        self.box_pub = rospy.Publisher('my_box3d', MarkerArray, queue_size=10)
        self.ego_pub = rospy.Publisher('my_ego_view', Marker, queue_size=10)

        self.rate = rospy.Rate(self.ros_rate)
        self.bridge = CvBridge()
        
        self.data_path = data_path
        dataset = pandaset.DataSet(self.data_path)
        print(dataset.sequences())
        
        self.seq_name = dataset.sequences()[seq_idx]
        self.data_seq = dataset[self.seq_name]
        self.data_seq.load()
        self.data_seq.lidar.set_sensor(lidar_idx)
        self.to_ego = to_ego
        
        self.camera_name = "front_camera"


    def Process(self, startfrom=0):
        frame_id = startfrom
        while not rospy.is_shutdown():

            # Publish camera-image
            if 1:
                image = self.data_seq.camera[self.camera_name][frame_id]
                image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                image = cv2.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3)))
                cv2.imshow("image", image)
                cv2.waitKey(1)
            
            # Publish camera field-angle 
            if self.field_angle > 0:
                publish_ego_car(self.ego_pub, self.field_angle)

            # # Publish point-cloud
            if 1:
                point_cloud = self.data_seq.lidar[frame_id].to_numpy()
                
                if self.to_ego:
                    point_cloud[:, :3] = pandaset.geometry.lidar_points_to_ego(point_cloud[:, :3], self.data_seq.lidar.poses[frame_id])
                
                # print(point_cloud.shape)
                publish_point_cloud(self.pcl_pub, point_cloud, down_ratio=1)

            if 1:
                frame_box_infos = FrameBoxData(is_waymo=True)
                frame_box_infos.from_pandaset(self.data_seq.cuboids[frame_id])
                # frame_box_infos.load_from_file(self.data_path + "/" + self.seq_name + "/preds/%d.pkl"%frame_id)
                frame_box_infos.filter_by_score(0.2)
                publish_3dbox(self.box_pub, frame_box_infos, None, 1.0/self.ros_rate)

            frame_id += 1
            rospy.loginfo("published [%d]"%(frame_id))
            self.rate.sleep()
            
            
if __name__ == "__main__":
    ppp = RvizPandasetPublisher(seq_idx = 55, lidar_idx = 1, to_ego=False, ros_rate=2)
    ppp.Process()