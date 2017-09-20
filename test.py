import cv2
import numpy as np
import math

cap = cv2.VideoCapture('test.mpg')
ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(gray)
snap = mask
center_old = 0
dist_fixed = [2,2,2,2,2,2,2]
dist_w = dist_fixed
weights = [1,1,1,1,3,4,5]
points = []

while(cap.isOpened()):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (3, 3), 0)
	_, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)

	image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
	if contours:
		cnt = max(contours, key = lambda x: cv2.contourArea(x))
		
		(x,y),radius = cv2.minEnclosingCircle(cnt)
		center_new = (int(x),int(y))
		
		if center_old:		
			#calculate point distance to estimate movement amount
			#compute a weighted average of dists
			dist = math.hypot(center_old[0] - center_new[0], center_old[1] - center_new[1])
			dist_w.pop()
			dist_w.insert(0, dist)
			
			if np.average(dist_w) >= 2.5 and dist < 50:
			
				cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
						
				#introduced an arrowedLine as a possible way to indicate direction
				#hopefully adding some additional differentiation for a NN to use 
				#as a distinguishing factor (and to filter out spurious gestures)
				#The arrowed line only draws at a modulo interval, so you get the
				#detailed curve interspersed with arrows
				
				#if center_old and spacing%2==0:
				#	cv2.arrowedLine(mask, center_old, center_new, (255,0,0), 3, 8, 0, 1)
				#	spacing += 1
				#elif not center_old:
				#	spacing = 0
				#else:
								
				cv2.line(mask, center_old, center_new, (255,255,255), 10)
				points.append(list(center_old))
				#spacing += 1
				
			else:
				
				ctr_points = np.array(points).reshape((-1,1,2)).astype(np.int32)
				
				image, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
				if contours:
					cnt = max(contours, key = lambda x: cv2.contourArea(x))
					x,y,w,h = cv2.boundingRect(cnt)
					
					if w*h > 40000:
						size = 10
						print(ctr_points)
						print(cnt)
						cv2.drawContours(snap, [ctr_points], 0, (255,255,255), size)
						
						snap = snap[y-size:(y+h+size),x-size:(x+w+size)]
						snap = cv2.resize(snap, (64,64))
						cv2.imshow('snap', snap)
				
				center_old = 0
				points=[]
				dist_w = dist_fixed
				mask = np.zeros_like(gray)
				snap = mask
					
		center_old=center_new
			

				
	cv2.imshow('frame',frame)
	cv2.imshow('thresh', thresh)
	cv2.imshow('mask', mask)
	#cv2.imshow('contours', img)
	#cv2,imshow('bounds', crop_img)
	
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break
	
	ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()

