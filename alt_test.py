import cv2
import numpy as np
import math
import time

def FindWandTip(image):
	p0 = []

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (3, 3), 0)
	ret, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
	
	ret, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	
	if len(contours):
		cnt = max(contours, key = lambda x: cv2.contourArea(x))
		(x,y),radius = cv2.minEnclosingCircle(cnt)
		p0 = np.array([x,y]).reshape((-1,1,2)).astype(np.float32)
	
	return p0
	
#TraceWandTip

#Some basic drawing variables used in the trace
size = 10
color = (255,255,255) #white

#Initialize the tracking arrays
#Dist_fixed is a reference for resetting the tracking
#Tracking can be adjusted by changing the number of tracked points
#Adding more points to the trace will balance out periods of slow movement
dist_thresh = 1.8
dist_fixed = [dist_thresh] * 10  
dist_w = dist_fixed[::]

#Tracking params
lk_params = dict( winSize  = (15,15),
			maxLevel = 2,
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#test capture with a saved video
cap = cv2.VideoCapture('test.mpg')

while True:
	#Read initial frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	mask = np.zeros_like(gray)
	dist_w = dist_fixed[::]
	
	#Find a likely wand tip
	p0 = FindWandTip(frame)
	
	#If we found one, perform tracking/traceing of that point
	while(len(p0)):		
		gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blur0 = cv2.GaussianBlur(gray0, (3, 3), 0)
		
		ret, frame = cap.read()
		gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blur1 = cv2.GaussianBlur(gray1, (3, 3), 0)

		p1, st, err = cv2.calcOpticalFlowPyrLK(gray0, gray1, p0, None, **lk_params)
		
		#Find good points
		good_new = p1[st==1]
		good_old = p0[st==1]
		
		if len(good_new):
			a,b = good_new[0].ravel()
			c,d = good_old[0].ravel()
			
			dist = math.hypot(c - a, d - b)
			dist_w.pop()
			dist_w.insert(0, dist)
			
			if np.average(dist_w) >= dist_thresh:			
				mask = cv2.line(mask, (a,b),(c,d), color, size)
			else:
				good_new=np.array([])
		
		cv2.imshow('framein', frame)
		cv2.imshow('maskin', mask)

		# Now update the previous frame and previous points
		print(dist_w)
		p0 = good_new.reshape(-1,1,2)
		
		if not(len(p0)):
			#This trace is over, right out the image
										
			#First, get the final drawn contour out of the mask image
			ret, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
			
			#If there was at least one, get the biggest one
			if contours:
				cnt = max(contours, key = lambda x: cv2.contourArea(x))
				
				#Get a bounding rectangle that we can use to measure and scale against
				x,y,w,h = cv2.boundingRect(cnt)
				
				#Another check for a good image, make sure its big enough to represent
				#an actual wand motion
				#play with the area as needed
				if w*h > 40000:
					#The mask image already contains our completed contour, so
					#crop and resize to fit the desired output
					
					#Crop the trace to fit the bounding box, with a buffer for the line width
					mask = mask[y-size:(y+h+size),x-size:(x+w+size)]
					#Then resize to 64x64 for optimization in the id routine
					mask = cv2.resize(mask, (64,64))
					#For debugging, show the trace output
					cv2.imshow('trace', mask)
					
					#Write a copy of the image to file
					#this can be modified later to cpature categorized images
					filename = 'images\\trace-' + str(time.time()).replace(".","-") + '.png'
					cv2.imwrite(filename, mask)
		
		
		k = cv2.waitKey(10) & 0xff
		if k == 27:
			#clean up
			cap.release()
			cv2.destroyAllWindows()
			exit()
		
	cv2.imshow('frame', frame)
	cv2.imshow('mask', mask)
	
	k = cv2.waitKey(10) & 0xff
	if k == 27:
		#clean up
		cap.release()
		cv2.destroyAllWindows()
		exit()
	

		
	

	








