# Copyright 2021 Toyota Research Institute.  All rights reserved.
import cv2
import numpy as np
# from tridet.structures.pose import Pose


class BEVImage:
    def __init__(
        self,
        metric_width=100.0,
        metric_height=100.0,
        pixels_per_meter=5,
        polar_step_size_meters=5,
        background_clr=(0, 0, 0)
    ):
        self._metric_width = metric_width
        self._metric_height = metric_height
        self._pixels_per_meter = pixels_per_meter
        self._polar_step_size_meters = polar_step_size_meters
        self._bg_clr = np.uint8(background_clr)[::-1].reshape(1, 1, 3)

        self._center_pixel = (int(metric_width * pixels_per_meter) // 2, int(metric_height * pixels_per_meter))
        self.reset()

        
    def __repr__(self):
        return 'width: {}, height: {}, data: {}'.format(self._metric_width, self._metric_height, type(self.data))

    
    def reset(self):
        """Reset the canvas to a blank image with guideline circles of various radii.
        """
        bev_height = int(self._metric_height * self._pixels_per_meter)
        bev_width = int(self._metric_width * self._pixels_per_meter)
        self.data = np.ones((bev_height, bev_width, 3),dtype=np.uint8) * self._bg_clr

        # Draw metric polar grid
        line_color = (75, 75, 75)
        zb_color = (150, 150, 150)
        font_type = cv2.FONT_HERSHEY_SIMPLEX
        font_size = 0.2
        thickness = 1
        lineType = cv2.LINE_AA
        cv2.line(self.data, (self._center_pixel[0], bev_height),(self._center_pixel[0], 0), line_color, thickness, lineType)
        cv2.putText(self.data, '0', (self._center_pixel[0]+1, bev_height-5), font_type, font_size, zb_color,1)  # draw center line
        for i in range(1, int(max(self._metric_width, self._metric_height)) // self._polar_step_size_meters):
            pixels_per_polar_step = int(i * self._polar_step_size_meters * self._pixels_per_meter)
            # draw grad
            cv2.line(self.data, (self._center_pixel[0]-pixels_per_polar_step, bev_height),(self._center_pixel[0]-pixels_per_polar_step, 0), line_color,thickness, lineType)
            cv2.line(self.data, (self._center_pixel[0]+pixels_per_polar_step, bev_height),(self._center_pixel[0]+pixels_per_polar_step, 0), line_color,thickness, lineType)
            cv2.line(self.data, (0, bev_height-pixels_per_polar_step),(bev_width, bev_height-pixels_per_polar_step), line_color,thickness, lineType)
            if int(i*self._polar_step_size_meters)%10==0:
                cv2.putText(self.data, str(int(i*self._polar_step_size_meters)), (self._center_pixel[0]-pixels_per_polar_step+1, bev_height-5), font_type, font_size, zb_color,1)
                cv2.putText(self.data, str(-int(i*self._polar_step_size_meters)), (self._center_pixel[0]+pixels_per_polar_step+1, bev_height-5), font_type, font_size, zb_color,1)
                cv2.putText(self.data, str(int(i*self._polar_step_size_meters)), (1, bev_height-pixels_per_polar_step-5), font_type, font_size, zb_color,1)

    @static
    def bev_box2corner()
                

    def render_bounding_box3d(self, bev_boxes, colors, thickness=3):
        """
        bev_boxes: N*5, [cx,cy,dx,dy,heading(x->y)]
        """
        if len(colors) == 1:
            colors = list(colors) * len(bev_boxes)
        boxes_corners = boxes3d.corners.cpu().numpy()
        # Draw cuboids
        for bidx, (corners, color) in enumerate(zip(boxes_corners, colors)):
            # Create 3 versions of colors for face coding.
            corners2d_x = corners[[0, 1, 5, 4], 0]
            corners2d_y = corners[[0, 1, 5, 4], 2]
            corners2d_x = self._center_pixel[0] + corners2d_x * self._pixels_per_meter
            corners2d_y = self._center_pixel[1] + corners2d_y * (-self._pixels_per_meter)
            # cv2.rectangle(self.data, (int(corners2d_x[3]), int(corners2d_y[3])), (int(corners2d_x[1]), int(corners2d_y[1])), color, thickness,lineType=cv2.LINE_AA)
            lineType = cv2.LINE_AA
            cv2.line(self.data, (int(corners2d_x[0]), int(corners2d_y[0])), (int(corners2d_x[1]), int(corners2d_y[1])), color, thickness, lineType)
            cv2.line(self.data, (int(corners2d_x[1]), int(corners2d_y[1])), (int(corners2d_x[2]), int(corners2d_y[2])),color, thickness, lineType)
            cv2.line(self.data, (int(corners2d_x[2]), int(corners2d_y[2])), (int(corners2d_x[3]), int(corners2d_y[3])),color, thickness, lineType)
            cv2.line(self.data, (int(corners2d_x[3]), int(corners2d_y[3])), (int(corners2d_x[0]), int(corners2d_y[0])), color, thickness, lineType)
            
            center_x = int(np.mean(corners2d_x))
            center_y = int(np.mean(corners2d_y))

            arrow_color = color # (0,0,255)
            front_face_x = (corners2d_x[0] + corners2d_x[1]) // 2
            front_face_y = (corners2d_y[0] + corners2d_y[1]) // 2
            front_top_x = int(center_x +  2 * (front_face_x - center_x))
            front_top_y = int(center_y +  2 * (front_face_y - center_y))
            cv2.arrowedLine(self.data, (center_x, center_y), (front_top_x, front_top_y), arrow_color, thickness)

    
    def save(self, save_path):
        if self.data is not None:
            cv2.imwrite(save_path, self.data)
        else:
            print("Bev data is none !")
    
            
if __name__ == "__main__":
    bev = BEVImage(metric_width=100.0, metric_height=100.0, pixels_per_meter=10, background_clr=(0, 0, 0))
    bev.save("a.jpg")
    
    