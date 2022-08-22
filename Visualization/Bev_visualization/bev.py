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

        self._center_pixel = (int(metric_width * pixels_per_meter) // 2, int(metric_height * pixels_per_meter) // 2)
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
        font_size = 0.5
        thickness = 1
        lineType = cv2.LINE_AA

        # draw horizonal center line
        cv2.arrowedLine(self.data, (self._center_pixel[0], bev_height),(self._center_pixel[0], 0), line_color, thickness+1, lineType, tipLength=0.03)
        cv2.putText(self.data, '0', (self._center_pixel[0]+1, bev_height-5), font_type, font_size, zb_color,1)

        # draw vertical center line
        cv2.arrowedLine(self.data, (bev_width, self._center_pixel[1]), (0, self._center_pixel[1]), line_color, thickness+1, lineType, tipLength=0.03)
        cv2.putText(self.data, '0', (0, self._center_pixel[1]+1), font_type, font_size, zb_color,1)

        # original point
        cv2.circle(self.data, self._center_pixel, 5, (0,255,255), thickness=-1)

        for i in range(1, int(self._metric_width) // self._polar_step_size_meters):
            pixels_per_polar_step = int(i * self._polar_step_size_meters * self._pixels_per_meter)
            cv2.line(self.data, (self._center_pixel[0]-pixels_per_polar_step, bev_height),(self._center_pixel[0]-pixels_per_polar_step, 0), line_color,thickness, lineType)
            cv2.line(self.data, (self._center_pixel[0]+pixels_per_polar_step, bev_height),(self._center_pixel[0]+pixels_per_polar_step, 0), line_color,thickness, lineType)
            if int(i*self._polar_step_size_meters)%10==0:
                cv2.putText(self.data, str(int(i*self._polar_step_size_meters)), (self._center_pixel[0]-pixels_per_polar_step+1, bev_height-5), font_type, font_size, zb_color,1)
                cv2.putText(self.data, str(-int(i*self._polar_step_size_meters)), (self._center_pixel[0]+pixels_per_polar_step+1, bev_height-5), font_type, font_size, zb_color,1)
        
        for i in range(1, int(self._metric_height) // self._polar_step_size_meters):
            pixels_per_polar_step = int(i * self._polar_step_size_meters * self._pixels_per_meter)
            cv2.line(self.data, (0, self._center_pixel[1]-pixels_per_polar_step),(bev_width, self._center_pixel[1]-pixels_per_polar_step), line_color,thickness, lineType)
            cv2.line(self.data, (0, self._center_pixel[1]+pixels_per_polar_step),(bev_width, self._center_pixel[1]+pixels_per_polar_step), line_color,thickness, lineType)
            if int(i*self._polar_step_size_meters)%10==0:
                cv2.putText(self.data, str(int(i*self._polar_step_size_meters)), (0, self._center_pixel[1]-pixels_per_polar_step+1), font_type, font_size, zb_color,1)
                cv2.putText(self.data, str(-int(i*self._polar_step_size_meters)), (0, self._center_pixel[1]+pixels_per_polar_step+1), font_type, font_size, zb_color,1)


    @staticmethod
    def bev_boxes2corners(bev_boxes):
        """
        bev_boxes: N*5, [cx,cy,dx,dy,heading(x->y)], np.array
        """
        bev_boxes = bev_boxes[:, [1,0,3,2,4]]
        dims = bev_boxes[:, [2,3]]    # dx <-> dy, for coord switch for opencv
        
        corners_origin = np.array(
            [[ 0.5,  0.5],
             [ 0.5, -0.5],
             [-0.5, -0.5],
             [-0.5,  0.5],]
        )
        corners_origin = np.expand_dims(corners_origin, 0).repeat(bev_boxes.shape[0], axis=0)
        corners = np.expand_dims(dims, 1) * corners_origin

        rot_sin = np.sin(bev_boxes[:, 4])
        rot_cos = np.cos(bev_boxes[:, 4])

        rot_mat_T = np.stack(
            [
                [rot_cos,  -rot_sin],
                [rot_sin,  rot_cos],
            ]
        )
        corners = np.einsum("aij,jka->aik", corners, rot_mat_T)
        corners += bev_boxes[:,:2].reshape([-1, 1, 2])
        return corners
     

    def render_bounding_bevbox(self, bev_boxes, colors, thickness=3):
        """
        bev_boxes: N*5, [cx,cy,dx,dy,heading(x->y)]
        """
        if len(colors) == 1:
            colors = list(colors) * len(bev_boxes)
        boxes_corners = BEVImage.bev_boxes2corners(bev_boxes)
        # Draw cuboids
        for bidx, (corners, color) in enumerate(zip(boxes_corners, colors)):
            # Create 3 versions of colors for face coding.
            corners2d_x = -corners[:, 0]
            corners2d_y = -corners[:, 1]
            corners2d_x = self._center_pixel[0] + corners2d_x * self._pixels_per_meter
            corners2d_y = self._center_pixel[1] + corners2d_y * self._pixels_per_meter
            # cv2.rectangle(self.data, (int(corners2d_x[3]), int(corners2d_y[3])), (int(corners2d_x[1]), int(corners2d_y[1])), color, thickness,lineType=cv2.LINE_AA)
            lineType = cv2.LINE_AA
            cv2.line(self.data, (int(corners2d_x[0]), int(corners2d_y[0])), (int(corners2d_x[1]), int(corners2d_y[1])), color, thickness, lineType)
            cv2.line(self.data, (int(corners2d_x[1]), int(corners2d_y[1])), (int(corners2d_x[2]), int(corners2d_y[2])),color, thickness, lineType)
            cv2.line(self.data, (int(corners2d_x[2]), int(corners2d_y[2])), (int(corners2d_x[3]), int(corners2d_y[3])),color, thickness, lineType)
            cv2.line(self.data, (int(corners2d_x[3]), int(corners2d_y[3])), (int(corners2d_x[0]), int(corners2d_y[0])), color, thickness, lineType)
            
            center_x = int(np.mean(corners2d_x))
            center_y = int(np.mean(corners2d_y))

            arrow_color = color # (0,0,255)
            front_face_x = (corners2d_x[0] + corners2d_x[3]) // 2
            front_face_y = (corners2d_y[0] + corners2d_y[3]) // 2
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

    boxes = np.array(
        [
            [30,15,8,2,0.5],
            [-40,-20,5,2.2, -np.pi/4.0],
        ]
    )
    bev.render_bounding_bevbox(boxes, colors=[(0,0,255)])
    bev.save("test.jpg")

    
    