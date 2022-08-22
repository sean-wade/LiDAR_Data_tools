import os
import cv2
import numpy as np
from bev import BEVImage


data = np.loadtxt("ddd.txt", delimiter=",")


boxes = data[:, [3,4,0,1,2]]
boxes[:, -1] = boxes[:, -1]/180*np.pi

print(boxes.shape)

bev = BEVImage(metric_width=100.0, metric_height=100.0, pixels_per_meter=10, background_clr=(0, 0, 0))

bev.render_bounding_bevbox(np.array(boxes), colors=[(0,0,255)])

cv2.imshow("r", bev.data)
cv2.waitKey(0)