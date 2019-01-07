import cv2
import numpy as np
import math
import sys
from config import WORK_DIR
#from bitarray import bitarray

cams = []
max_cams = 255
cons = np.zeros((0,0), dtype = np.bool)
max_dist = 10
selected_cam = None
placing = True
invalidInput = True
floorplan_file = WORK_DIR + "METADATA/" + "flrpln.png"
org_img = cv2.imread(floorplan_file)
cam_img = org_img.copy()
conn_img = cam_img.copy()


def dist(p1,p2): return (math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))

def closest_cam(x, y):
	closestCam = [-1,math.inf]
	for c in range(len(cams)):
		d = dist((x,y),(cams[c][0],cams[c][1]))
		if(d<closestCam[1]): closestCam = [c,d]
	if(closestCam!=-1 and closestCam[1]<max_dist): return closestCam[0]
	else: return None

def update_connections_img():
	global conn_img
	conn_img = cam_img.copy()
	if(len(cons)>0):
		for row in range(len(cons)):
			for c in range(row,len(cons)):
				if(cons[row][c]):
					cv2.line(conn_img,(cams[row][0],cams[row][1]), (cams[c][0],cams[c][1]), (0,0,0), 4)

def update_cameras_img():
	global cam_img
	cam_img = org_img.copy()
	for c in cams:
		cv2.circle(cam_img, (c[0],c[1]),5,(0,0,255),-1)

def mouse_evt(event,x,y,flags,param):
	global cons,cams, selected_cam
	if(event == cv2.EVENT_LBUTTONDOWN):
		if(placing):
			if(len(cams)<max_cams):
				cams.append((x,y))
				cons = np.concatenate((cons,np.zeros((1,len(cons)),dtype = np.bool)), axis=0)
				cons = np.concatenate((cons, np.zeros((len(cons), 1),dtype = np.bool)),axis=1)
				update_cameras_img()
				cv2.imshow("Floorplan",cam_img)
		elif(not placing):
			indx = closest_cam(x,y)
			if(indx != None):
				if(selected_cam == None):
					selected_cam = indx
				elif(selected_cam != None):
					if(indx != selected_cam):
						cons[indx][selected_cam] = not cons[indx][selected_cam]
						cons[selected_cam][indx] = not cons[selected_cam][indx]
						selected_cam = None
						update_connections_img()
						cv2.imshow("Floorplan",conn_img)
					elif(indx == selected_cam):
						selected_cam = None

	elif(event == cv2.EVENT_MOUSEMOVE and placing == False and selected_cam != None):
		cur_img = conn_img.copy()
		cv2.line(cur_img, (cams[selected_cam][0],cams[selected_cam][1]), (x,y), (0,0,0), 4)
		cv2.imshow("Floorplan", cur_img)
	elif(event == cv2.EVENT_MBUTTONDOWN):
		if(placing):
			if(len(cams)>0):
				indx = closest_cam(x,y)
				if(indx != None):
					cons = np.delete(cons, indx, axis = 0)
					cons = np.delete(cons, indx, axis = 1)
					del cams[indx]
					update_cameras_img()
					cv2.imshow("Floorplan",cam_img)

while(invalidInput):
	inpt = input("Do you want to overwrite any existing information?(y/n) ")
	if(inpt == "y"):
		#clear files
		f = open("camera_placement.txt","w")
		f.close()
		f = open("connection_info.bin","wb")
		f.close()
		invalidInput = False
	elif(inpt == "n"):
		try:
			f_cams = open("camera_placement.txt","r")
			f_cons = open("connection_info.txt","r")
			f_cams.seek(0)
			f_cons.seek(0)
			for x in f_cams:
				x = x.rstrip()
				if(x[0] == "("):
					vals = (x[1:len(x)-1]).split(",")
					for v in range(len(vals)):
						vals[v] = int(vals[v])
					cams.append(tuple(vals))
					
			update_cameras_img()

			cons = np.zeros((len(cams),len(cams)),dtype=np.bool)
			itr = 0
			for x in f_cons:
				q = x.rstrip()
				for j in range(len(q)):
					cons[itr][j] = bool(int(q[j]))
				itr += 1
				#cons[itr] = list(x)
			update_connections_img()
			# with f_cons as f:
			# 	first_byte = f.read(1)
			# 	num_cams = int.frombytes(byte,byteorder='big')
			# 	cons = np.array((num_cams,num_cams),dtype=np.bool)
			# 	bytes_per_cam = math.ceil(num_cams/8)
			# 	byte = f.read(bytes_per_cam)
			# 	itr = 0
			# 	while byte:
			# 		a = bitarray()
			# 		a.frombytes(byte)
			# 		cons[itr] = (np.frombuffer(a.unpack(zero=b'\x00',one='\xFF'),dtype=np.bool,count=-1))[:num_cams]
			# 		itr += 1
			# 		byte = f.read(bytes_per_cam)
		except: pass
		invalidInput = False
print("Number of cameras must be "+str(max_cams)+" or less")
print("Use Esc key to save data and close the window")
print("Use Spacebar to enter Camera Placement mode")
print("Use Enter key to enter Camera Connetion mode")

cv2.namedWindow("Floorplan")
cv2.setMouseCallback("Floorplan",mouse_evt)
cv2.imshow("Floorplan",cam_img)
while(True):
	k = cv2.waitKey(1) & 0xff
	if k == 27:
		if(len(cams)>0):
			f_cams = open(WORK_DIR + "METADATA/" + "camera_placement.txt","w")
			for x in range(len(cams)):
				f_cams.write(str(cams[x])+"\n")
			f_cams.close()
			f_cons = open(WORK_DIR + "METADATA/" + "connection_info.txt","w")
			for x in range(len(cons)):
				for j in range(len(cons[x])):
					f_cons.write(str(int(cons[x][j])))
				f_cons.write("\n")
			f_cons.close()
		cv2.destroyAllWindows()
		print("Ending...")
		sys.exit()
	elif k == 32:
		if(not placing):
			print("Placement Mode")
			update_cameras_img()
			cv2.imshow("Floorplan",cam_img)
			placing = True
	elif k == 13:
		if(placing):
			print("Connection Mode")
			update_connections_img()
			cv2.imshow("Floorplan",conn_img)
			placing = False





