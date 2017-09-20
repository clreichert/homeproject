import cv2
import numpy as np
import math
import time


#flag to indicate if we are actively tracing a capture
tracing = False
#Also, when we start a new trace, we have no existing points
center_old = 0

#Some basic drawing variables used in the trace
size = 10
color = (255,255,255) #white

#Initialize the tracking arrays
#Dist_fixed is a reference for resetting the tracking
#Tracking can be adjusted by changing the number of tracked points
#Adding more points to the trace will balance out periods of slow movement
#--consider using deque for performance gains here--
dist_fixed = [2,2,2,2,2,2,2]
dist_w = dist_fixed

#Weights can be used to further tune the trace equation, but I found them
#to be fickle
#weights = [1,1,1,1,3,4,5]

#test capture with a saved video
cap = cv2.VideoCapture('test.mpg')

#run a video capture loop
while(cap.isOpened()):
	#read a frame of video
	ret, frame = cap.read()
	
	#perform basic image transformations: grayscale, blur, thresholding
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (3, 3), 0)
	_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
	
	#if starting a new trace, set up our mask, and reset the tracking variables
	if not(tracing):
		mask = np.zeros_like(gray)
		trace = mask
		center_old = 0
		dist_w = dist_fixed
		tracing = True
		
	#find contours in the image
	image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	
	#if there are any, inspect them and decide what to do
	if contours:
	
		#take the single largest contour that is visible
		#ideally, this will find a wand tip or other IR reflective object
		#that is close to the camera and ignore other stuff
		#--consider changing this to pick the contour that is closest to the previous center point--
		cnt = max(contours, key = lambda x: cv2.contourArea(x))
		
		#find the approximate center of the identified contour, this is our new point
		(x,y),radius = cv2.minEnclosingCircle(cnt)
		center_new = (int(x),int(y))
		
		#if we already have a prior point, we can do some math and add to our trace
		if center_old:		
			#calculate point distance to estimate the amount of movement
			#add this distance to our tracking array and pop off the oldest value
			dist = math.hypot(center_old[0] - center_new[0], center_old[1] - center_new[1])
			dist_w.pop()
			dist_w.insert(0, dist)
			
			#Look at both the average over the last few points and the current distance
			#If they conform to expectations, this can be included in our trace
			#The Average is intended to reset the trace when the wand is 'hovering'
			#The Total Distance is intended to prevent 'blips' from throwing off the trace
			if np.average(dist_w) >= 2.5 and dist < 50:
			
				#This is debugging code, it shows the detected contour in the original image
				cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
				
				#This draws a trace line on our mask image
				cv2.line(mask, center_old, center_new, (255,255,255), 10)
			
			#When we fail the average or distance checks, we have reached the end of a trace
			#so we do some processing on the trace and reset it
			else:
				
				#Performance note, should be able to improve speed by replacing the findCountours
				#call below with a routine that accumulates a list of contour points during the
				#trace and then converts that to a contour array, this function kind of works:
				#
				#	ctr_points = np.array(points).reshape((-1,1,2)).astype(np.int32)
				#
				#but it connects the first and last points for some reason, worth looking at if
				#Pi performance is an issue
				
				#First, get the final drawn contour out of the mask image
				image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
				
				#If there was at least one, get the biggest one
				if contours:
					cnt = max(contours, key = lambda x: cv2.contourArea(x))
					
					#Get a bounding rectangle that we can use to measure and scale against
					x,y,w,h = cv2.boundingRect(cnt)
					
					#Another check for a good image, make sure its big enough to represent
					#an actual wand motion
					#play with the area as needed
					if w*h > 40000:
					
						#Draw the final trace onto our output
						cv2.drawContours(trace, [cnt], 0, (255,255,255), size)
						
						#Since the trace is initially the same as our mask, the bounding box
						#still applies
						#Crop the trace to fit the bounding box, with a buffer for the line width
						trace = trace[y-size:(y+h+size),x-size:(x+w+size)]
						#The resize to 64x64 for optimization in the id routine
						trace = cv2.resize(trace, (64,64))
						#For debugging, show the trace output
						cv2.imshow('trace', trace)
						
						#sample image writing code
						filename = 'images\\trace-' + str(time.time()).replace(".","-") + '.png'
						cv2.imwrite(filename, trace)

				
				#Since the trace is over, reset the tracing flag
				tracing=False
		
		#After doing everything, always move the points along
		#--again, could maybe obsolete this by just accumulating an array of contour points--
		center_old=center_new
			
	#For debugging, show the image in various states			
	cv2.imshow('frame',frame)
	cv2.imshow('thresh', thresh)
	cv2.imshow('mask', mask)
	
	#Exit clause
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
	
#clean up
cap.release()
cv2.destroyAllWindows()

