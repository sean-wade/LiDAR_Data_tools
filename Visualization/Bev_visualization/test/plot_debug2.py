import os
import cv2
import numpy as np
from bev import BEVImage


root = "/mnt/data/waymo_opensets/val/for_cpp/debug/preds_global/"

fs = ["seq_100_frame_%d.txt"%ii for ii in range(198)]



for ff in fs:
    world_boxes = np.loadtxt(root+ff)
    
    world_boxes = world_boxes[:, [0,1,3,4,6]]

    # world_boxes[:,:2] += np.array([1.70363128e+03  ,-1.44607796e+04])
    # world_boxes[:,:2] += np.array([2.08400451e+03,-1.26923412e+04])
    world_boxes[:,:2] += np.array([2.53272184e+04  ,-4.19410244e+04])

    print(world_boxes[0])


    bev = BEVImage(metric_width=200.0, metric_height=200.0, pixels_per_meter=10, background_clr=(0, 0, 0))

    bev.render_bounding_bevbox(world_boxes, colors=[(0,0,255)])
    # cv2.imshow("r", bev.data)
    # cv2.waitKey(0)
    bev.save("res_waymo/%s.jpg"%ff[:-4])

    

    


