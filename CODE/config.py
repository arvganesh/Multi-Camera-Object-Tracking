import sys
import math
import os
import time

WORK_DIR = os.path.normpath(os.getcwd() + os.sep + os.pardir) + "/"
SSD_PATH = WORK_DIR + "DEPENDENCIES/SSD"
REID_PATH = WORK_DIR + "DEPENDENCIES/deeppersonreid"
VIDEO_PATH = WORK_DIR + "VIDEOS"
PATH_TO_WEIGHTS = SSD_PATH + "/weights/ssd300_mAP_77.43_v2.pth"
CAM_PLACEMENT_PATH = WORK_DIR + "METADATA/camera_placement.txt"
TRACKFILE_PATH = WORK_DIR + "/METADATA/" + "trackfile.txt"

# print ("WD", WORK_DIR)
# print ("SP", SSD_PATH)
# print ("RP", REID_PATH)
# print ("VP", VIDEO_PATH)
# print ("WP", PATH_TO_WEIGHTS)
# print ("CAM", CAM_PLACEMENT_PATH)