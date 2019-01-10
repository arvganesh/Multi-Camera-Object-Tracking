import time
import sys
import cv2
from cv2 import selectROI
from config import *
from collections import namedtuple
from deep_reid import deep_reid
import track_logger as track_logger
from initial_detect import init_det
import splicer
from time_analyze import Analyzer
Camera_info = namedtuple('Camera_info', ['frame_rate', 'start_frame', 'connections', 'file_name'])

# print("Fix DIM_X and DIM_Y in config. I put place holders but they arent the right values")

t = Analyzer()

""" INIT MODELS """
net = build_ssd('test', 300, 21) # init SSD
net.load_weights(PATH_TO_WEIGHTS) # init SSD

model = models.init_model("resnet50", num_classes=1, loss={'xent'}, use_gpu=False) # init REID

""" *********** """

def check_bboxes(frame):
	# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	global subject
	bboxes_ppl = init_det(frame=frame, net=net)
	for bbox in bboxes_ppl:
		bbox_img_org = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
		bbox_img = cv2.resize(bbox_img_org, (SEM_DIM_X,SEM_DIM_Y))
		# attrs = semantic_attribute_det(bbox_img)
		# if (attrs == subject["attr_id"]):
		# 	#bbox_img = cv2.resize(frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]], (REID_DIM_X,REID_DIM_Y))

		# while True:
		# 	cv2.imshow("image", bbox_img_org)
		# 	cv2.imshow("img2", subject["img"])
		# 	k = cv2.waitKey(1) & 0xff
		# 	if k == 27:
		# 		cv2.destroyAllWindows()
		# 		break
		# 		# sys.exit()
		dist = deep_reid(subject["img"], bbox_img_org, model)
		# print ("Distmat", dist)
		if(dist < DIST_THRESHOLD):
			# print ("bbox",bbox)
			return bbox
	return None
				
#clear trackfile **chnage to hdf5 if necisary
f = open(WORK_DIR + "METADATA/" + "trackfile.txt", "w")
f.close()

#get inputs
st = time.time()
# start_time = int(input("Start Time: "))
# end_time = int(input("End Time: "))
# data_inc = int(input("Data increment: "))
# start_camera_id = int(input("Start Camera ID: "))

start_time = 24
end_time = 70
data_inc = 2
start_camera_id = 0


#set time diff to check against
#end_time -= start_time
# if(end_time <= 0):
	# print("End time is before Start time. MAIN")
	# sys.exit()


try:
	f_cons = open(WORK_DIR + "METADATA/" + "connection_info.txt","r")
	f_cons.seek(0)
	cons = []
	itr = 0
	for x in f_cons:
		cons.append([])
		q = x.rstrip()
		for j in range(len(q)):
			if(bool(int(q[j]))): cons[itr].append(j)
		itr += 1
except:
	print("Error opening connection config file. MAIN")
	sys.exit()


#set camera info
#Unhard code this **
camera_video_files = [0]*len(cons)

camera_video_files[0] = Camera_info(frame_rate = 8,start_frame = 10*8,file_name = VIDEO_PATH+"/cam_8.avi",connections = cons[0])
camera_video_files[1] = Camera_info(frame_rate = 9,start_frame = 2*9,file_name = VIDEO_PATH+"/cam_30.avi",connections = cons[1])
camera_video_files[2] = Camera_info(frame_rate = 30,start_frame = 3*10,file_name = VIDEO_PATH+"/cam_23.avi",connections = cons[2])
camera_video_files[3] = Camera_info(frame_rate = 21,start_frame = 44*21,file_name = VIDEO_PATH+"/cam_10.avi",connections = cons[3])

#initialize caps
caps = []
for x in range(len(camera_video_files)):
	caps.append(cv2.VideoCapture(camera_video_files[x].file_name))
	if(not caps[x].isOpened()):
		print("Error opening video. MAIN")
		sys.exit()
	caps[x].set(5, camera_video_files[x].frame_rate)
	caps[x].set(1,max(camera_video_files[x].start_frame-1,0))
	ok,frame = caps[x].read()
	if not ok:
		print("Cannot read video file "+str(camera_video_files[x].file_name)+". MAIN")
		sys.exit()

cur_time = start_time
cur_cam_id = start_camera_id

ok,frame = caps[start_camera_id].read()

#bbox = (348, 80, 53, 124)
bbox  = selectROI(frame,False)
cv2.destroyAllWindows()

subject = {}


# bbox_img = cv2.resize(frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]], (REID_DIM_X,REID_DIM_Y))
bbox_img = frame[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]]
# print("bbox_img type",type(bbox_img))
subject["img"] = bbox_img
subject["bbox"] = bbox
# subject["attr_id"] = semantic_attribute_det(bbox_img)

TOTAL_TRACKLOG_TIME = 0
TOTAL_REID_TIME = 0

while(cur_time < end_time):
	print("Now using camera"+str(cur_cam_id))
	stt = time.time()
	frames_elapsed = track_logger.track(cam_id = cur_cam_id, cap = caps[cur_cam_id], bbox = subject["bbox"], data_inc = data_inc, time_analyzer = t)
	ett = time.time()
	TOTAL_TRACKLOG_TIME += ett-stt

	cur_time += int(frames_elapsed/camera_video_files[cur_cam_id].frame_rate)
	surrounding_cams = camera_video_files[cur_cam_id].connections
	if(len(surrounding_cams) == 0):
		print("Cam_"+str(cur_cam_id)+" does not have any surrounding cameras. MAIN")
		print ("Amount of video analyzed: "+str(cur_time) +" seconds.")
		break
	begin_search_time = time.time()
	found = False
	frame_itr = 0
	frame_itr_interval = REID_NUM_SKIP_FRAMES
	print("Calculating",end="",flush=True)
	stt = time.time()
	while(not found):
		if(time.time()-begin_search_time > MAX_REID_TIME):
			print("Could not RE-ID subject within "+str(MAX_REID_TIME)+" seconds. Ending Program.  MAIN")
			cur_time = end_time+1 #break outer while
			break
		for cam in surrounding_cams:
			ok, frame = caps[cam].read()
			if(not ok):
				print("Not ok frame. MAIN")
			else:
				print(".",end="",flush=True)
				match_bbox = check_bboxes(frame)
				if(match_bbox != None):
					subject["bbox"] = match_bbox
					cur_cam_id = cam
					found = True
					print ("")
					break
		
			for _ in range(frame_itr_interval):
				caps[cam].read()
		frame_itr += frame_itr_interval
	ett = time.time()
	TOTAL_REID_TIME += ett-stt
	if(cur_cam_id == 0): break #0 is last camera so finish search	

et = time.time()
print("TOTAL_TRACKLOG_TIME",TOTAL_TRACKLOG_TIME)
print("TOTAL_REID_TIME",TOTAL_REID_TIME)
print("\n")
print("  **************** Summary ****************")
print("      Total Time: {}       ".format(str(et-st)))
print("            Algorithm Completed           ")
print("  *****************************************")
print("\n")

t.show_times()

invalid_input = True
while(invalid_input):
	inpt = input("Would you like to generate the video? [y/n] ")
	if(inpt == "y"):
		splicer.splicer(camera_video_files,TRACKFILE_PATH,data_inc)
		print("Done generating video. Closing Program.  MAIN")
		invalid_input = False
	elif(inpt == "n"):
		print("Ok. Closing Program")
		invalid_input = False
