import sys
import math
import os
import time
import cv2
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as torch_trans



WORK_DIR = os.path.normpath(os.getcwd() + os.sep + os.pardir) + "/"
SSD_PATH = WORK_DIR + "DEPENDENCIES/SSD"
REID_PATH = WORK_DIR + "DEPENDENCIES/deeppersonreid"
VIDEO_PATH = WORK_DIR + "VIDEOS"
PATH_TO_WEIGHTS = SSD_PATH + "/weights/ssd300_mAP_77.43_v2.pth"
CAM_PLACEMENT_PATH = WORK_DIR + "METADATA/camera_placement.txt"
TRACKFILE_PATH = WORK_DIR + "/METADATA/" + "trackfile.txt"
DIST_THRESHOLD = 260
MAX_REID_TIME = 30
REID_NUM_SKIP_FRAMES = 5
REID_DIM_X = 128
REID_DIM_Y = 256
SEM_DIM_X = 1
SEM_DIM_Y = 1
MAX_TRACK_FRAMES = 150
MAX_TRACK_ERROR_FRAMES = 5

sys.path.append(SSD_PATH)
sys.path.insert(0, (WORK_DIR + 'DEPENDENCIES/deeppersonreid'))

from ssd import build_ssd
from torchreid import models

# print ("WD", WORK_DIR)
# print ("SP", SSD_PATH)
# print ("RP", REID_PATH)
# print ("VP", VIDEO_PATH)
# print ("WP", PATH_TO_WEIGHTS)
# print ("CAM", CAM_PLACEMENT_PATH)