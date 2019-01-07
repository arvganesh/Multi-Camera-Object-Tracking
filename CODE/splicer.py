from config import *
#cam_files as array of all video files from cameras
#data_frame_increment is number of frames skipped between each data point

#splicer.splicer(camera_video_files,TRACKFILE_PATH,data_inc)
def splicer(camera_video_files, TRACKFILE_PATH, data_inc):
	f = open(TRACKFILE_PATH,"r")
	caps = []
	for x in range(len(cam_files)):
		# print ("camfiles:", x)
		caps.append(cv2.VideoCapture(VIDEO_PATH + "/" + camera_video_files[x].file_name))
		caps[len(caps)-1].set(cv2.CAP_PROP_FPS, camera_video_files[x].frame_rate)
		#caps[x] = (cv2.VideoCapture(cam_files[x]))
		#caps[x].set(cv2.CAP_PROP_FPS, frame_rate)

	bbox_w = 0
	bbox_h = 0
	for x in f:
		x = x.rstrip("\n")
		# print("X:", x)
		if(x[0]=="!"):
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
			for z in range(data_frame_increment):
				#time.sleep(0.1)
				ok, frame = caps[cam_ID_to_use].read()
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
splicer([[0,"cam_8.avi",8],[1,"cam_30.avi",9],[2,"cam_23.avi",30],[3,"cam_10.avi",21]],WORK_DIR + "METADATA/" + "trackfile.txt", 9, 3)