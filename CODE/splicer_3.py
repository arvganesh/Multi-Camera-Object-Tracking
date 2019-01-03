import cv2 
from cv2 import selectROI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import math
import time
#cam_files as array of all video files from cameras
#data_frame_increment is number of frames skipped between each data point
def index_by_ID(a,ID,func):
	for i in range(len(a)):
		if(func(a[i]) == ID): return i
	return -1
def splicer(cam_files,log_file, frame_rate, data_frame_increment):
	f = open(log_file,"r")
	cam_files.sort(key = lambda x: x[0])
	caps = []
	for x in range(len(cam_files)):
		# print ("camfiles:", x)
		caps.append(cv2.VideoCapture(cam_files[x][1]))
		caps[len(caps)-1].set(cv2.CAP_PROP_FPS, cam_files[x][2])
		#caps[x] = (cv2.VideoCapture(cam_files[x]))
		#caps[x].set(cv2.CAP_PROP_FPS, frame_rate)

	bbox_w = 0
	bbox_h = 0
	for x in f:
		x = x.rstrip("\n")
		# print("X:", x)
		if(x==""): continue
		elif(x[0]=="!"):
			vals = x[2:len(x)-1].split(",")
			cam_ID_to_use = int(vals[0])
			print("Using Cam_"+str(cam_ID_to_use))
			cam_cap_index = index_by_ID(cam_files, cam_ID_to_use, lambda x: x[0])
			caps[cam_cap_index].set(cv2.CAP_PROP_POS_FRAMES,int(vals[1]))
			# print("startfrm: ",int(vals[1]))
			bbox_w = int(vals[2])
			bbox_h = int(vals[3])
		elif(x[0] == "-"): continue
		else:
			bbox = x[1:len(x)-1].split(",")
			p1 = (int(float(bbox[0])), int(float(bbox[1])))
			p2 = (int(float(bbox[0]) + bbox_w), int(float(bbox[1]) + bbox_h))
			for z in range(data_frame_increment):
				#time.sleep(0.1)
				ok, frame = caps[cam_cap_index].read()
				if not ok: 
					break
					#return False
				else:
					cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
					cv2.imshow("Frame",frame)
				k = cv2.waitKey(30) & 0xff
				if k == 27 : break
				# Exit if ESC pressed
				
	f.close()

#uncomment to use as stand-alone file	
#splicer([[0,"cam_8.avi",8],[3,"cam_30.avi",9],[1,"cam_23.avi",30],[2,"cam_10.avi",21]],"trackfile.txt", 9, 3)