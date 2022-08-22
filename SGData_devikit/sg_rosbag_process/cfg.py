import os
_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep

"""
    Each type of topic with its folder_name to create.
    Do not edit the values, just add/comment if needed !
""" 
default_topic_sensor_name_dict = {
    "camera" : {
        "/sensor/camera/f30/compressed"         : "f30",
        "/sensor/camera/f60/compressed"         : "f60",
        "/sensor/camera/f120/compressed"        : "f120",
        # "/sensor/camera/fl120/compressed"       : "fl120",
        # "/sensor/camera/fr120/compressed"       : "fr120",
        # "/sensor/camera/rl120/compressed"       : "rl120",
        # "/sensor/camera/rr120/compressed"       : "rr120",
        # "/sensor/camera/r120/compressed"        : "r120", 
    },

    "lidar" : {
        "/sensor/lidar/undistortion" : "lidar_undist",
        # "/sensor/lidar/fusion"       : "lidar",
    },

    "imu" : {
        "/sensor/ins/rawimu" : "imu"
    },

    "ins" : {
        "/sensor/ins/fusion" : "ins"
    }
}


# the calibration file. ralated to the car. must be same with the rosbag.
default_calibration_files = {
    "/sensor/camera/f30/compressed"   : _path + "MKZ-A3QV50/cameraf30_intrinsics.yaml",
    "/sensor/camera/f60/compressed"   : _path + "MKZ-A3QV50/cameraf60_intrinsics.yaml",
    "/sensor/camera/f120/compressed"  : _path + "MKZ-A3QV50/cameraf120_intrinsics.yaml",
}

