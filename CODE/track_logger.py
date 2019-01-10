import cv2 
from config import WORK_DIR, MAX_TRACK_FRAMES, MAX_TRACK_ERROR_FRAMES, VIDEO_PATH
from cv2 import selectROI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import math
import time

'''
import h5py
with h5py.File('random.hdf5', 'w') as f:
    dset = f.create_dataset("default", data=arr)
    dset = f.create_dataset("yeet", data=arr[:5])
    f.close()

with h5py.File('random.hdf5', 'r') as f:
    deff = f['default']
    yeet = f['yeet']
    print(deff, yeet)
    f.close()
with h5py.File('PreprocessedData.h5', 'w') as hf:
    hf.create_dataset("X_train", data=X_train_data, maxshape=(None, 512, 512, 9))


'''

#frames_elapsed, error = track_logger.track(cap = caps[cur_cam_indx], bbox = bbox, data_inc = data_inc)
def track(cam_id, cap, bbox, data_inc, time_analyzer):
    f = open((WORK_DIR + "/METADATA/" + "trackfile.txt"),"a+")
    track_info = []
    frm_cnt = 0
    print("Progress Cam_" + str(cam_id) + ": ", end="", flush=True)
    #cam, start frame, bbox width, bbox height
    f.write("\n"+"!("+str(cam_id)+","+str(int(cap.get(1)))+","+str(bbox[2])+","+str(bbox[3])+")\n")
    f.write("("+str(int(bbox[0]))+","+str(int(bbox[1]))+"]")
    # track_info.append("\n"+"!("+str(cam_id)+","+str(int(cap.get(1)))+","+str(bbox[2])+","+str(bbox[3])+")\n")
    # track_info.append("["+str(int(bbox[0]))+","+str(int(bbox[1]))+"]")

    tracker_type = "CSRT"
    # if tracker_type == 'BOOSTING':
    #         tracker = cv2.TrackerBoosting_create()
    # if tracker_type == 'MIL':
    #     tracker = cv2.TrackerMIL_create()
    # if tracker_type == 'KCF':
    #     tracker = cv2.TrackerKCF_create()
    # if tracker_type == 'TLD':
    #     tracker = cv2.TrackerTLD_create()
    # if tracker_type == 'MEDIANFLOW':
    #     tracker = cv2.TrackerMedianFlow_create()
    # if tracker_type == 'GOTURN':
    #     tracker = cv2.TrackerGOTURN_create()
    # if tracker_type == 'MOSSE':
    #     tracker = cv2.TrackerMOSSE_create()
    # if tracker_type == "CSRT":'
    # st = time.time()
    tracker = cv2.TrackerCSRT_create()
    # print ("tracker time", time.time() - st)
    """
    BOOSTING -> ye
    MIL -> H E C K no
    KCF -> ehhhhhhhhhhhhhh
    TLD -> straight trash lmao
    MEDIANFLOW -> see TLD
    GOTURN -> nope
    MOSSE -> see above and TLD
    CSRT -> yeh
    """
    ok, frame = cap.read()
    # st = time.time()

    ok = tracker.init(frame, tuple(bbox))
    # print ("tracker time", time.time() - st)
    fail_detect_itrs = 0
    cycle = 0
    while True:
        # Read a new frame
        time_analyzer.start("reading")
        ok, frame = cap.read()
        time_analyzer.stop("reading")
        frm_cnt += 1
        if not ok:
            f.close()
            # write_to_trackfile(track_info)
            #finish out progress bar
            numofthing = math.ceil((MAX_TRACK_FRAMES - frm_cnt) / data_inc)
            print("#" * numofthing, end="", flush=True)
            print("")
            # with h5py.File((WORK_DIR + "/METADATA/" + "trackfile.hdf5"), 'w') as f:
            #     f.create_dataset("tracklog", data=track_info)
            #     f.close()
            return frm_cnt # goes to next cams

        # Update tracker
        time_analyzer.start("update")
        ok, bbox = tracker.update(frame)
        time_analyzer.stop("update")
        
        #only record at data_inc
        if(cycle==data_inc-1):
            cycle=0
            #record bbox pos
            if ok:
                # track_info.append([int(bbox[0]),int(bbox[1])])
                f.write("("+str(int(bbox[0]))+","+str(int(bbox[1]))+")\n")
            else:
                # track_info.append([-1,-1])
                f.write("-e-\n")
            #increment progress bar
            print("#", end="", flush=True)
        else:
            cycle+=1
            if(not ok):
                fail_detect_itrs+=1
                if(fail_detect_itrs > MAX_TRACK_ERROR_FRAMES):
                    # write_to_trackfile(track_info)
                    f.close()
                    print("")
                    return frm_cnt
        if(frm_cnt > MAX_TRACK_FRAMES):
            # write_to_trackfile(track_info)
            f.close()
            print("")
            return frm_cnt
        


def write_to_trackfile(track_info):
    # st = time.time()
    with open(WORK_DIR + "/METADATA/" + "trackfile.txt", "a+") as f:
        for item in track_info:
            f.write(str(item) + "\n")
            # print(str(item) + "\n")
        f.close()
    # et = time.time()
    # print ("write time:", et - st)
    # sys.exit()
    
#uncomment to use as stand-alone file
# start_frame = 8*3
# vid = VIDEO_PATH + "/cam_8.avi"
# cap = cv2.VideoCapture(vid)
# cap.set(cv2.CAP_PROP_FPS, 8)
# cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
# ok, frame = cap.read()
# bbox = selectROI(frame)
# track(0,cap,bbox,2)


# cam_id, cap, bbox, data_inc

"""
What we know
- bounding box coords are accurate
- tracker is activated, but for some reason stuck in the same spot.
"""