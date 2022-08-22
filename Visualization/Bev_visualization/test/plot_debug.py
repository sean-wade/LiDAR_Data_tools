import os
import cv2
import numpy as np
from bev import BEVImage


root = "/mnt/data/SGData/download/25_debug/"

fs = os.listdir(root)

os.makedirs("res/local", exist_ok=True)
os.makedirs("res/world", exist_ok=True)

for ff in fs:

    all_data = open(root+ff, "r").read().split("\n\n")

    local_boxes = all_data[0].strip().split("\n")
    local_boxes_num = []
    for wb in local_boxes:
        wb_data = wb.split(",")
        # print(wb_data)
        x,y,dx,dy,h = float(wb_data[1]), float(wb_data[2]), float(wb_data[4]), float(wb_data[5]), float(wb_data[7])
        local_boxes_num.append([x,y,dx,dy,h])

    print(local_boxes_num)
    lb_bev = BEVImage(metric_width=200.0, metric_height=200.0, pixels_per_meter=10, background_clr=(0, 0, 0))

    lb_bev.render_bounding_bevbox(np.array(local_boxes_num), colors=[(0,0,255)])
    lb_bev.save("res/local/%s.jpg"%ff[:-4])


    world_boxes = all_data[1].strip().split("\n")
    world_boxes_num = []
    for wb in world_boxes:
        wb_data = wb.split(",")
        # print(wb_data)
        x,y,dx,dy,h = float(wb_data[1]), float(wb_data[2]), float(wb_data[4]), float(wb_data[5]), float(wb_data[7])
        world_boxes_num.append([x,y,dx,dy,h])

    print(world_boxes_num)
    wb_bev = BEVImage(metric_width=200.0, metric_height=200.0, pixels_per_meter=10, background_clr=(0, 0, 0))

    wb_bev.render_bounding_bevbox(np.array(world_boxes_num), colors=[(0,0,255)])
    # cv2.imshow("r", bev.data)
    # cv2.waitKey(0)
    wb_bev.save("res/world/%s.jpg"%ff[:-4])

    

    


