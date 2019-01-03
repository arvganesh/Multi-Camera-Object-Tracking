import cv2 
from cv2 import selectROI
import numpy as np
from config import WORK_DIR, CAM_PLACEMENT_PATH, SSD_PATH, VIDEO_PATH, TRACKFILE_PATH
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import math
import os
import time
f = open((TRACKFILE_PATH), "w")
f.close()
import sys
import traceback

# class TracePrints(object):
#   def __init__(self):    
#     self.stdout = sys.stdout
#   def write(self, s):
#     self.stdout.write("Writing %r\n" % s)
#     traceback.print_stack(file=self.stdout)

# sys.stdout = TracePrints()
import time
import track_logger_3 as track_logger
import splicer_3 as splicer
from evaluate_cams import person_reid as re_id
from initial_detect import init_det


# init det returns what re-id needs. 

st = time.time()
start_time = 0
end_time = 56
dist_thresh = 260
#fps = 30
start_camera = 0
data_inc = 3
#cam id, cam file, frame rate
camera_video_files = [[0,"cam_8.avi",8],[2,"cam_10.avi",21],[1,"cam_23.avi",30],[3,"cam_30.avi",9]]
start_frames = [10*8,44*21,int((5/3)*30),int(18)]
#camera_video_files = ["Athletics-Mens-100m-T44-Final-London-2012-Paralympic-Games.mp4"]
#building_floorplan_file = "flrpln.png"

def index_by_ID(a,ID,func):
	for i in range(len(a)):
		if(func(a[i]) == ID): return i
	return -1

connections = []
cams = []
#org_cam_img = cv2.imread(building_floorplan_file)
#cv2.namedWindow("Floorplan")

try:
	f = open(CAM_PLACEMENT_PATH,"r+")
except IOError:
	f = open(CAM_PLACEMENT_PATH,"w+")
f.seek(0)
reading_cams = True
for x in f:
	x = x.rstrip()
	if(x==""): continue
	elif x[0]=="-":
		#print("broke")
		reading_cams = False
	elif x[0]!="!" and reading_cams:
		vals = x[1:len(x)-1].split(",")
		for val in range(len(vals)): vals[val] = int(vals[val])
		cams.append(tuple(vals))
	elif reading_cams==False:
		comma_index = x.index(",")
		close_bracket_index = x.index("]")
		vals = [x[1:comma_index],x[comma_index+3:close_bracket_index].split(",")]
		vals[0] = int(vals[0])
		for v in range(len(vals[1])): vals[1][v] = int(vals[1][v])
		#print("VALS: "+str(vals))
		connections.append(vals)

#cam_img = org_cam_img.copy()
invalidInput = False
f.close()


start_camera_indx = index_by_ID(camera_video_files, start_camera, lambda x: x[0])
cur_time = 0
start_frame = start_frames[start_camera_indx]
#start_frame = int(start_time*camera_video_files[start_camera_indx][2])

#end_frame = int(end_time*fps)
#cur_frm += start_frame
print (VIDEO_PATH + "/" + camera_video_files[start_camera_indx][1])
cap = cv2.VideoCapture(VIDEO_PATH + "/" + camera_video_files[start_camera_indx][1])
cap.set(cv2.CAP_PROP_FPS, camera_video_files[start_camera_indx][2])
cap.set(1, start_frame)
if (not cap.isOpened()):
	print("Error opening video. main")
	sys.exit()
ok, frame = cap.read()
if not ok:
	print("Cannot read video file")
	sys.exit()


# bbox = selectROI(frame, False) # !!!!!!
bbox = (348, 80, 53, 124)
# bbox = (72, 120, 101, 184)
frm2 = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
frm2 = cv2.resize(frm2, (128, 256))
cv2.imwrite("/home/arvganesh/Documents/SciFair18-19/deeppersonreid/data/single_test/query/0005_c5s1_555555_00.jpg", frm2)

#video,cam,start_frame,bbox,data_inc,frame_rate

track_info = [camera_video_files[start_camera][1],start_camera,start_frame,bbox,data_inc,camera_video_files[start_camera_indx][2]]
while(cur_time < end_time):
	print("Now using camera "+str(track_info[1]))
	frames_elapsed, error = track_logger.track(VIDEO_PATH + "/" + track_info[0],track_info[1],track_info[2],track_info[3],track_info[4],track_info[5])
	#print("track log: ",frames_elapsed,error)
	cur_time += int(frames_elapsed/camera_video_files[index_by_ID(camera_video_files,track_info[1],lambda x: x[0])][2])
	#print("curtm",cur_time)
	#if(error != "LOST_SUBJECT"): print("Error: "+str(error))

	surrounding_cams = connections[index_by_ID(connections, track_info[1], lambda x: x[0])][1]
	#surrounding_cams.append(track_info[1])   
	if(surrounding_cams[0]==0):
		#print("circle")
		break
	caps = [0]*len(surrounding_cams)
	for x in range(len(surrounding_cams)):
		cam_indx = index_by_ID(camera_video_files, surrounding_cams[x], lambda x: x[0])
		caps[x] = (cv2.VideoCapture(VIDEO_PATH + "/" + camera_video_files[cam_indx][1]))
		#print("Checking vid: ",camera_video_files[cam_indx][1])
		caps[x].set(cv2.CAP_PROP_FPS, camera_video_files[cam_indx][2])
		caps[x].set(1,start_frames[cam_indx])
		#print("stfrm: ",start_frames[cam_indx])
		# if(track_info[1] == 3):
		# 	ok,frm = caps[x].read()
		# 	cv2.imshow("frame",frm)
		# 	while True:
		# 		k = cv2.waitKey(30) & 0xff
	done = False
	frame_itr = 0
	frame_itr_interval = 5
	print ("Calculating", end="", flush=True)
	while(not done):
		count = 0
		for x in range(len(caps)):
			ok, frame = caps[x].read()
			if(not ok): 
				print("not ok frame (main)")
			else:
				# count += 1
				# count %= 3
				# print("count: "+str(count))
				print (".", end="", flush=True)
				match_bbox = None
				bboxes_ppl = init_det(frame=frame) # person detection (pytorch)
				#print("bb_ppl", bboxes_ppl)
				c = 0
				for bbox in bboxes_ppl:
					frm2 = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
					frm2 = cv2.resize(frm2, (128, 256))
					cv2.imwrite("/home/arvganesh/Documents/SciFair18-19/deeppersonreid/data/single_test/bounding_box_test/0001_c2s1_123456_00.jpg", frm2)
					

					match = re_id() # reid model
					if (match[0][0] < dist_thresh): 
						match_bbox = bbox
						# print("frame: ",frame_itr)
						# print("bbox: ",bbox)
						break
				#match is bounding box of matching person. Should return None otherwise
				if(match_bbox != None):
					print ("\n")
					done = True
					cam_indx = index_by_ID(camera_video_files, surrounding_cams[x],lambda x: x[0])
					track_info = [camera_video_files[cam_indx][1],surrounding_cams[x],start_frames[cam_indx]+frame_itr,match_bbox,data_inc,camera_video_files[cam_indx][2]]
					# freeze frame w/ bbox on camera X
					# if(track_info[1] == 3):
					# 	p1 = int(match_bbox[0]), int(match_bbox[1])
					# 	p2 = int(match_bbox[0] + match_bbox[2]), int(match_bbox[1] + match_bbox[3])
					# 	cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
					# 	cv2.imshow("frame",frame)
					# 	while True:
					# 		k = cv2.waitKey(30) & 0xff
					break
			frame_itr += frame_itr_interval
			# print("added 5 frames")
			for _ in range(frame_itr_interval-1):
				caps[x].read()
			
			
	if(track_info[1] == 0):
		# print("DONEDONE")
		break
et = time.time()

print("\n")
print("  **************** Summary ****************")
print("      Total Time: {}       ".format(str(et-st)))
print("            Algorithm Completed           ")
print("  *****************************************")
print("\n")
# print("Total Time: "+)
# print("Algorithm Completed.")
invalid_input = True
while(invalid_input):
	inpt = input("Would you like to see the video? [y/n] ")
	if(inpt == "y"):
		splicer.splicer(camera_video_files,TRACKFILE_PATH,30,data_inc)
	elif(inpt!="n"): invalid_input=True
	else: invalid_input=False

'''
try:
	f = open("camera_placement.txt","r+")
except IOError:
	f = open("camera_placement.txt","w+")
f.seek(0)
reading_cams = True
for x in f:
	x = x.rstrip()
	if(x==""): continue
	elif x[0]=="-":
		#print("broke")
		reading_cams = False
	elif x[0]!="!" and reading_cams:
		vals = x[1:len(x)-1].split(",")
		for val in range(len(vals)): vals[val] = int(vals[val])
		cams.append(tuple(vals))
	elif reading_cams==False:
		comma_index = x.index(",")
		close_bracket_index = x.index("]")
		vals = [x[1:comma_index],x[comma_index+3:close_bracket_index].split(",")]
		vals[0] = int(vals[0])
		for v in range(len(vals[1])): vals[1][v] = int(vals[1][v])
		#print("VALS: "+str(vals))
		connections.append(vals)
for c in cams:
	cv2.circle(cam_img,(c[1],c[2]),5,(0,0,255),-1)
cv2.imshow("Floorplan",cam_img)
invalidInput = False
f.close()
'''
