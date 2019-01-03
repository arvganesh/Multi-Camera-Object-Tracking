import cv2 
from cv2 import selectROI
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
import math
from config import CAM_PLACEMENT_PATH
import time

def dist(p1,p2): return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
def index_by_camID(cams,ID):
	i = min(len(cams)-1,ID)
	while(cams[i][0] != ID and i > -1): i -= 1
	return i
def closest_cam_index(cams_,x,y,max_dist):
	closestCam = [-1,math.inf]
	for c in range(len(cams_)):
		d = dist((x,y),(cams_[c][1],cams_[c][2]))
		if(d<closestCam[1]):
			closestCam = [c,d]
	if(closestCam[0]!=-1 and closestCam[1]<max_dist): return closestCam[0]
	else: return None
def draw_connections(cams, connections, cam_img):
	con = []
	global conn_img
	conn_img = cam_img.copy()
	for c in connections:
		indx1 = index_by_camID(cams, c[0])
		for c2 in c[1]:
			if(not c2 in con):
				indx2 = index_by_camID(cams, c2)
				cv2.line(conn_img,(cams[indx1][1],cams[indx1][2]),(cams[indx2][1],cams[indx2][2]),(0,0,0),4)
		con.append(c)
	#cv2.imshow("Floorplan",conn_img)
def mouse_evt(event, x, y, flags, param):
	global cams, connections, cam_img
	if(event == cv2.EVENT_LBUTTONDOWN):
		if(placing):
			global cam, cam_img
			if(cam<=cam_max):
				info = (cam,x,y)
				cv2.circle(cam_img,(x,y),5,(0,0,255),-1)
				cv2.imshow("Floorplan",cam_img)
				print("Camera_"+str(cam)+" placed")
				cams.append(info)
				cam += 1
			if(cam>cam_max):
				print("You can't add anymore.")
		elif(not placing):
			
			indx = closest_cam_index(cams,x,y,10)
			if(indx != None):
				global selected_cam
				if(selected_cam == None):
					selected_cam = cams[indx][0]
					print("Selecting Camera_"+str(selected_cam))
				elif(selected_cam != None):
					if(cams[indx][0]!=selected_cam):
						#indx1 is closest cam index in connections
						indx1 = len(connections)-1
						if(indx1 !=-1):
							while(connections[indx1][0] != cams[indx][0] and indx1 >-1): indx1 -= 1
						#indx2 is selected_cam index in connections
						indx2 = len(connections)-1
						if indx2 != -1:
							while(connections[indx2][0] != selected_cam and indx2 > -1): indx2 -= 1
						# print("Closest id: "+str(cams[indx][0]))
						# print("connections: "+str(connections))
						# print("Indx1: "+str(indx1))
						# print("Indx2: "+str(indx2))
						
						#print("indx1 after while: "+str(indx1))
						#indx1 = connections.find(selected_cam)
						
						#print("indx2 after while: "+str(indx2))
						#indx2 = connections.find(cams[indx][0])
						global conn_img
						if(indx1 != -1 and indx2 != -1 and selected_cam in connections[indx1][1]):
							print("Disconnecting cameras: "+str(selected_cam)+" and "+str(cams[indx][0]))
							del connections[indx1][1][connections[indx1][1].index(selected_cam)]
							del connections[indx2][1][connections[indx2][1].index(cams[indx][0])]
							conn_img = cam_img.copy()
							draw_connections(cams,connections,cam_img)
							cv2.imshow("Floorplan", conn_img)
							#print("Showing conn_img")
						else:
							print("Connecting cameras: "+str(selected_cam)+" and "+str(cams[indx][0]))
							if(indx1 == -1):
								#print("cam_"+str(cams[indx][0])+" not in connections")
								connections.append([cams[indx][0],[selected_cam]])
								
							else:
								connections[indx1][1].append(selected_cam)
								
							if(indx2 == -1):
								#print("cam_"+str(cams[indx][0])+" not in connections")
								connections.append([selected_cam,[cams[indx][0]]])
							else:
								connections[indx2][1].append(cams[indx][0])
						
							sel_cam_index = index_by_camID(cams, selected_cam)
							cv2.line(conn_img, (cams[sel_cam_index][1],cams[sel_cam_index][2]), (cams[indx][1],cams[indx][2]), (0,0,0), 4)
							cv2.imshow("Floorplan",conn_img)
						#i = min(selected_cam,len(cams)-1)
						#while(cams[i][0] != selected_cam and i > -1): i-=1
						#cams[i][3].append(cams[indx][0])
						#cams[indx][3].append(selected_cam)
						#print("Connections: "+str(connections))
						#print("New cams: "+str(cams))
					else:
						print("Unselecting Camera_"+str(selected_cam))
					selected_cam = None
	elif(event == cv2.EVENT_MOUSEMOVE and placing==False and selected_cam != None):
		img_f = conn_img.copy()
		i = index_by_camID(cams,selected_cam)
		#i = min(selected_cam,len(cams)-1)
		#while(cams[i][0] != selected_cam and i>0): i -= 1
		cv2.line(img_f,(cams[i][1],cams[i][2]),(x,y),(0,0,0),4)
		cv2.imshow("Floorplan",img_f)
	elif(event == cv2.EVENT_MBUTTONDOWN):
		if(placing):
			if(len(cams)>0):
				indx = closest_cam_index(cams,x,y,10)
				if indx != None:
					print("Removing Camera_"+str(cams[indx][0]))
					#print("Cam: "+str(cams[indx]))
					indx1 = len(connections)-1
					if(len(connections)>0):
						while(connections[indx1][0] != cams[indx][0] and indx1 >-1): indx1 -= 1
					#indx1 = connections.find()
					if(indx1 != -1):
						#print("indx1: "+str(indx1))
						#print("connections: "+str(connections))
						for c in connections[indx1][1]:
							indx2 = len(connections)-1
							if(len(connections)>0):
								while(connections[indx2][0] !=  c and indx2 >-1): indx2 -= 1
							#print("index of "+str(c)+" is "+str(indx2))
							if(indx2 != -1): del connections[indx2][1][connections[indx2][1].index(cams[indx][0])]
						del connections[indx1]
					#del cams[i][3][cams[i][3].index(cams[indx][0])]
					del cams[indx]
					
					#print("New cams: "+str(cams))
					cam_img = org_img.copy()
					for c in cams:
						cv2.circle(cam_img,(c[1],c[2]),5,(0,0,255),-1)
					cv2.imshow("Floorplan",cam_img)


#cv2.imshow("Floorplan",img)
#p_img = org_img.copy()

if __name__ == '__main__':

	selected_cam = None
	placing = True
	invalidInput = True
	cams = []
	connections = []
	cam_max = 999
	building_floorplan_file = "flrpln.png"
	org_img = cv2.imread(building_floorplan_file)
	cam_img = org_img.copy()
	conn_img = org_img.copy()

	while(invalidInput):
		inpt = input("Do you want to overwrite any existing information?(y/n) ")
		if(inpt == "y"):
			#Overwrite file
			f = open(CAM_PLACEMENT_PATH,"w")
			f.close()
			invalidInput = False
		if(inpt == "n"):
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
			for c in cams:
				cv2.circle(cam_img,(c[1],c[2]),5,(0,0,255),-1)
			cv2.imshow("Floorplan",cam_img)
			invalidInput = False
			f.close()
			#print("Cams: "+str(cams))


	if(len(cams)>0):
		cam = cams[len(cams)-1][0]+1
	else: cam = 0
	#print("cams: "+str(cams))

	print("Number of cameras must be "+str(cam_max)+" or less")
	print("Use Esc key to save data and close the window")
	print("Use Spacebar to enter Camera Placement mode")
	print("Use Enter key to enter Camera Connetion mode")

	cv2.namedWindow("Floorplan")
	cv2.setMouseCallback("Floorplan",mouse_evt)
	cv2.imshow("Floorplan",cam_img)
	while(True):
		#cv2.imshow("Floorplan",img)
		# Exit if ESC pressed
		k = cv2.waitKey(1) & 0xff
		if k == 27 :
			#print("cams: "+str(cams))
			f = open("camera_placement.txt","w")
			f.write("!"+"X"*((len(str(cam_max)))-(len(str(len(cams)))))+str(len(cams))+"\n")
			for c in cams:
				f.write(str(c)+"\n")
			if(len(connections)>0):
				f.write("---\n")
			#print("connections before sort: "+str(connections))
			for c in range(len(connections)-1,-1,-1):
				#print("C: "+str(c))
				#print("connections: "+str(connections[c][1]))
				if(len(connections[c][1])==0): del connections[c] 
			connections.sort(key = lambda x: x[0])
			for c in connections:
				f.write(str(c)+"\n")
			#print("connections after sort: "+str(connections))
			f.close()
			cv2.destroyAllWindows()
			print("Ending...")
			break
		#Space
		elif k == 32:
			if(not placing):
				print("Placement Mode")
				cv2.imshow("Floorplan",cam_img)
				placing = True
		#Enter
		elif k == 13:
			if(placing):
				print("Connetion Mode")
				conn_img = cam_img.copy()
				draw_connections(cams,connections,cam_img)
				cv2.imshow("Floorplan",conn_img)
				placing = False
			