from config import *

Camera_info = namedtuple('Camera_info', ['frame_rate', 'start_frame', 'connections', 'file_name'])
#cam_files as array of all video files from cameras
#data_frame_increment is number of frames skipped between each data point

#splicer.splicer(camera_video_files,TRACKFILE_PATH,data_inc)
def splicer(camera_video_files, TRACKFILE_PATH, data_inc):
	f = open(TRACKFILE_PATH, "r")
	caps = []
	for x in range(len(camera_video_files)):
		# print ("camfiles:", x)
		caps.append(cv2.VideoCapture(camera_video_files[x].file_name))
		# print (VIDEO_PATH + "/" + camera_video_files[x].file_name)
		caps[len(caps)-1].set(cv2.CAP_PROP_FPS, camera_video_files[x].frame_rate)
		#caps[x] = (cv2.VideoCapture(cam_files[x]))
		#caps[x].set(cv2.CAP_PROP_FPS, frame_rate)

	bbox_w = 0
	bbox_h = 0
	for x in f:
		x = x.rstrip("\n")
		if(x == ""): continue
		elif(x[0]=="!"):
			vals = x[2:len(x)-1].split(",")
			cam_ID_to_use = int(vals[0])
			print("Using Cam_"+str(cam_ID_to_use))
			caps[cam_ID_to_use].set(cv2.CAP_PROP_POS_FRAMES,int(vals[1]))
			# print("startfrm: ",int(vals[1]))
			bbox_w = int(vals[2])
			bbox_h = int(vals[3])
		elif(x[0] == "["):
			bbox = x[1:len(x)-1].split(",")
			p1 = (int(float(bbox[0])), int(float(bbox[1])))
			p2 = (int(float(bbox[0]) + bbox_w), int(float(bbox[1]) + bbox_h))
			for z in range(data_inc):
				#time.sleep(0.1)
				ok, frame = caps[cam_ID_to_use].read()

				if not ok: 
					print ("not ok. SPLICER")
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

# camera_video_files = [0]*4

# cons = [0]*4

# camera_video_files[0] = Camera_info(frame_rate = 8,start_frame = 10*8,file_name = VIDEO_PATH+"/cam_8.avi",connections = cons[0])
# camera_video_files[1] = Camera_info(frame_rate = 9,start_frame = 2*9,file_name = VIDEO_PATH+"/cam_30.avi",connections = cons[1])
# camera_video_files[2] = Camera_info(frame_rate = 30,start_frame = 3*10,file_name = VIDEO_PATH+"/cam_23.avi",connections = cons[2])
# camera_video_files[3] = Camera_info(frame_rate = 21,start_frame = 44*21,file_name = VIDEO_PATH+"/cam_10.avi",connections = cons[3])

# splicer(camera_video_files,WORK_DIR + "METADATA/" + "trackfile.txt", 2)