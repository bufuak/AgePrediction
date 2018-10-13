from matplotlib import pyplot as plt
import numpy as np
import cv2 as cv
import os
import time
import csv
from PIL import Image 
import numpy
import shutil


def detect_face(image):
    face_cascade = cv.CascadeClassifier('.\\Documents\\opencv-3.4.1\\data\\haarcascades_cuda\\haarcascade_frontalface_default.xml')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = []
    faces = face_cascade.detectMultiScale(gray,
	scaleFactor=1.01,
	minNeighbors=5,
	minSize=(20,20))

    xf,yf,wf,hf,bf = 0,0,0,0,0
    for (x,y,h,w) in faces:
    	if bf < w*h:
    		xf,yf,wf,hf = x,y,w,h
        
    sub_face = image[yf:yf+wf, xf:xf+hf]

    return sub_face


def function(dir):
	directory = os.fsencode(dir)
	errors = []
	noface = 0
	moreface = 0
	minSizes = [(20,20)]
	scaleFactors = [1.01]

	results = []

	for scale in scaleFactors:
		for minSize in minSizes:
			start_time = time.time()
			noface = 0
			moreface = 0
			filecount = 0
			for file in os.listdir(directory):
			    filename = os.fsdecode(file)
			    if filename.endswith(".png") or filename.endswith(".jpg"): 
			        path = ".\\" + dir + "\\" + filename
			        print(path)
			        img = cv.imread(path)
			        print("detecting face")
			        face = detect_face(img)
			        print("detected")
			        if len(face) != 0:
			        	scaledface = cv.resize(face,(256,256))
			        else:
			        	scaledface = cv.resize(img,(256,256))
			        
			        cv.imwrite('.\\'+ dir +'_haar\\' + filename ,scaledface)
			        os.rename(path, '.\\' + dir +'2\\' + filename)
			        print(directory)

	
function("valid")






