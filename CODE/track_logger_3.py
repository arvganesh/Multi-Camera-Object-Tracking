import cv2 
from config import WORK_DIR
from cv2 import selectROI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import math
import time

def track(video,cam,start_frame,bbox,data_inc,frame_rate):
    MAX_FRAME = 150
    # print("args: ",video,cam,start_frame,bbox,data_inc,frame_rate)
    f = open((WORK_DIR + "/METADATA/" + "trackfile.txt"),"a+")
    frm_cnt = 0
    print("Progress Cam_" + str(cam) + ": ", end="", flush=True)
    #cam, start frame, bbox width, bbox height
    f.write("\n"+"!("+str(cam)+","+str(start_frame)+","+str(bbox[2])+","+str(bbox[3])+")\n")
    f.write("("+str(int(bbox[0]))+","+str(int(bbox[1]))+")\n")
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_FPS, frame_rate)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if (not cap.isOpened()):
        print("Error opening video from Track Logger.")
        sys.exit()
    ok, frame = cap.read()
    # print("type: "+str(type(frame)))
    frm_cnt += 1
    if not ok:
        print("Cannot read video file")
        sys.exit()

    tracker_type = "CSRT"
    if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()

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
    ok = tracker.init(frame,tuple(bbox))

    fail_detect_itrs = 0
    cycle = 0
    while True:
        # Read a new frame
        ok, frame = cap.read()
        frm_cnt += 1
        if not ok:
            f.close()
            # print ("Frame count", frm_cnt)
            numofthing = math.ceil((MAX_FRAME - frm_cnt) / data_inc)
            print("#" * numofthing, end="", flush=True)
            print("")
            return frm_cnt,"BAD_FRAME" # goes to next cams

        # Update tracker
        ok, bbox = tracker.update(frame)
        #if not ok: print("not found")

        
        if(cycle==data_inc-1):
            cycle=0
            if ok:
                f.write("("+str(int(bbox[0]))+","+str(int(bbox[1]))+")\n")
            else:
                f.write("-e-\n")
            print("#", end="", flush=True)
        else:
            cycle+=1
            if(not ok):
                fail_detect_itrs+=1
                if(fail_detect_itrs>5):
                    f.close()
                    print("")
                    return frm_cnt,"LOST_SUBJECT"
        if(frm_cnt > MAX_FRAME):
            f.close()
            print("")
            return frm_cnt, "TOO_MANY_FRAMES"
        # cv2.waitKey(int(1000/(frame_rate/2)))
        # # Exit if ESC pressed
        # k = cv2.waitKey(1) & 0xff
        # if k == 27 :
        #     f.close()
        #     return frm_cnt, "WAIT_KEY"

#uncomment to use as stand-alone file
# start_frame = 9*2
# vid = "cam_30.avi"
# cap = cv2.VideoCapture(vid)
# cap.set(cv2.CAP_PROP_FPS, 9)
# cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
# ok, frame = cap.read()
# bbox = selectROI(frame)
# track("cam_30.avi",3,start_frame,bbox,3,9)




"""
What we know
- bounding box coords are accurate
- tracker is activated, but for some reason stuck in the same spot.
"""