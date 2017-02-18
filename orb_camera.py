import numpy as np
import cv2

cap = cv2.VideoCapture(0) #capture video from camera

orb = cv2.ORB_create()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

GREEN = (0, 155, 0)

previousDes = None
previousKp = None
previousFrame = None

while(1):
	ret, frame = cap.read()
	kp, des = orb.detectAndCompute(frame, None)
	if previousDes is not None:
		matches = bf.match(previousDes,des)
		matches = sorted(matches, key = lambda x:x.distance)
		print len(matches)
		cv2.drawMatches(previousFrame,previousKp,frame,kp,matches, frame)
	previousDes = des
	previousFrame = frame
	previousKp = kp
	#third argument is output image. I put frame because I had no idea what other thing to pass here. 
	# seems to work though
	cv2.drawKeypoints(frame,kp,frame, color=GREEN, flags=0)
	cv2.imshow('orb_features',frame)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

