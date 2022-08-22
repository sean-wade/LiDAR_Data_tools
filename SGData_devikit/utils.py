import cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
    

############################################################
### Data statistics plot.
############################################################

def plot_dict(data_dict, title="", save_path=None):

    data = list(data_dict.values())
    labels = list(data_dict.keys())
    
    plt.figure(figsize=(12,8))    
    plt.bar(range(len(data)), data)
    plt.xticks(range(len(data)),labels)
    plt.xlabel(title)
    plt.ylabel("Count number")
    plt.title(title + " distribution")

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_list(data, title="", save_path=None):
    plt.figure(figsize=(12,8))
    sns.distplot(data, kde=True, rug=False, bins=100)
    plt.title(title + " distribution")

    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

############################################################


############################################################
### Pointcloud to BEV-RGB image.
############################################################

def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


def point_cloud_2_birdseye(points,
                           res=0.1,
                           side_range=(-100., 100.),  # left-most to right-most
                           fwd_range = (-100., 100.), # back-most to forward-most
                           height_range=(-2., 2.),  # bottom-most to upper-most
                           save_path = None
                           ):
    """ Creates an 2D birds eye view representation of the point cloud data.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        res:        (float)
                    Desired resolution in metres to use. Each output pixel will
                    represent an square region res x res in size.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the origin.
                    All height values will be clipped to this min and max value,
                    such that anything below min will be truncated to min, and
                    the same for values above max.
    Returns:
        2D numpy array representing an image of the birds eye view.
    """
    # EXTRACT THE POINTS FOR EACH AXIS
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    intensity_points = points[:, 3]

    # FILTER - To return only indices of points within desired cube
    # Three filters for: Front-to-back, side-to-side, and height ranges
    # Note left side is positive y axis in LIDAR coordinates
    f_filt = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    s_filt = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()

    # KEEPERS
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    intensity_points = intensity_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32)  # x axis is -y in LIDAR
    y_img = (-x_points / res).astype(np.int32)  # y axis is -x in LIDAR

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor & ceil used to prevent anything being rounded to below 0 after shift
    x_img -= int(np.floor(side_range[0] / res))
    y_img += int(np.ceil(fwd_range[1] / res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a=z_points,
                           a_min=height_range[0],
                           a_max=height_range[1])

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values = scale_to_255(pixel_values,
                                min=height_range[0],
                                max=height_range[1])

    intensity_points = scale_to_255(intensity_points,
                                    min=255,
                                    max=0)

    # INITIALIZE EMPTY ARRAY - of the dimensions we want
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max, 3], dtype=np.uint8)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    pixel_values = pixel_values.repeat(3, axis=0).reshape(-1,3)
    pixel_values[:,1] = intensity_points
    im[y_img, x_img] = pixel_values
    
    return im


def plot_sgobjs_on_image(image, labels, res=0.1, side_range=(-100., 100.), fwd_range = (-100., 100.)):
    """
        labels: sg-json-file format objects
    """
    # height, width, _ = image.shape
    for obj in labels["objects"]:
        txt1 = "%s_%s"%(obj["class"], obj["track_id"])
        txt2 = "%s_%s"%(obj["attributes"].get("occlusion", ""), obj["attributes"].get("ignore", ""))
        
        cy = -(obj["bbox"]["center_x"]) / res
        cx = -(obj["bbox"]["center_y"]) / res
        dy = int((obj["bbox"]["length"]) / res)
        dx = int((obj["bbox"]["width"])  / res)
        heading = (-obj["bbox"]["heading"]) / np.pi * 180

        cx -= int(np.floor(side_range[0] / res))
        cy += int(np.ceil(fwd_range[1] / res))

        cx, cy = int(cx), int(cy)

        # global_speed = [0, 0]

        rot_rectangle = ((cx, cy), (dx, dy), heading)
        box = cv2.boxPoints(rot_rectangle) 
        box = np.int0(box) #Convert into integer values
        image = cv2.drawContours(image, [box], 0, (0,128,255), 3)
        cv2.putText(image, txt1, (cx, cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0,140,255),1)
        cv2.putText(image, txt2, (cx, cy+10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (106,106,255),1)
    
    return image
    

############################################################
