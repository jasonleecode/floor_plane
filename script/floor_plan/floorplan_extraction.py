import cv2, time
import numpy as np
from skimage.morphology import medial_axis

from floor_plan.utils import *

# 简单的LaserScan类，替代ROS依赖
class LaserScan:
    def __init__(self):
        self.header = None
        self.angle_min = 0.0
        self.angle_max = 0.0
        self.angle_increment = 0.0
        self.range_min = 0.0
        self.range_max = 0.0
        self.ranges = []

RESIZE_RATIO = 5
DISTANCE_THRESHOLD = 5
LASER_RESOLUTION = 0.2

def bitmap_to_graph(bitmap):
    med4, dist = medial_axis_four(bitmap, return_distance=True)
    dist_mask = (dist < DISTANCE_THRESHOLD)*(dist!=0)

    med4_pruned = prune_end_point(med4, dist_mask)
    verts = find_vertices(med4_pruned)
    edges = extract_edges(verts,med4_pruned)

    return verts, edges, dist

def load_floor_plan(file):
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    image = np.pad(image,10)
    _, bitmap = cv2.threshold(image, 250, 1, cv2.THRESH_BINARY)
    _, bitmap = cv2.threshold(cv2.resize(bitmap*255, 
                                    (int(bitmap.shape[1]/RESIZE_RATIO), 
                                    int(bitmap.shape[0]/RESIZE_RATIO)), 
                                    interpolation=cv2.INTER_LINEAR), 
                                    0, 
                                    1, 
                                    cv2.THRESH_BINARY)

    return bitmap

# def process_floor_plan(floor_plan):
    
#     nodes, edges = bitmap_to_graph(floor_plan) #return edges in the form of (origin,destination,path_len)
#     return nodes, edges

def get_laserscan(dist_map, point, resolution):

    ranges = laserscan_sim_raymarching(dist_map, point, resolution=resolution)/10.0
    scan = LaserScan()
    scan.header.frame_id = "map"

    # we reverse the angle to reverse the y axis.
    scan.angle_max = -2*np.pi
    scan.angle_min = 0.0
    scan.range_min = 0.01
    scan.range_max = 100.0
    scan.angle_increment = (scan.angle_max-scan.angle_min)/ranges.shape[0] 
    scan.ranges = ranges

    return scan


    