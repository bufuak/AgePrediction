import numpy as np
import cv2 as cv
import os
import time
from PIL import Image 
import numpy
import csv
import shutil


def detect_face(image):
    face_cascade = cv.CascadeClassifier('.\\Documents\\opencv-3.4.1\\data\\haarcascades\\haarcascade_frontalface_default.xml')
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

def detect_face_deep(image):
	save = image
	print("[INFO] loading model...")
	net = cv.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it
	(h, w) = image.shape[:2]
	blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()
	bestConfidence = 0.0
	bestsX,bestsY,besteX,besteY = 0,0,0,0
	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > 0.5:
			if confidence>bestConfidence:
				bestConfidence = confidence
				# compute the (x, y)-coordinates of the bounding box for the
				# object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				bestsX,bestsY,besteX,besteY = startX,startY,endX,endY

	# if(bestsX >= 300 or bestsY >=300 or besteX >=300 or besteY >= 300):
	# 	return save
	print(besteX,h,besteY,w)
	if besteX>w or besteY>h or besteX<=0 or besteY<=0:
		sub_face = save
		returnValue = 0
	else:
		sub_face = image[bestsY:besteY, bestsX:besteX]
		returnValue = 1
	return sub_face,returnValue


def cropAllWithDeep(dir,destination,backup):
	start_time = time.time()
	directory = os.fsencode(dir)
	f = open(dir+".txt","a+")
	filecount = 0
	for file in os.listdir(directory):
		filecount += 1
		filename = os.fsdecode(file)
		print(filename)
		if filename.endswith(".png") or filename.endswith(".jpg"): 
			path = ".\\" + dir + "\\" + filename
			print(path)
			img = cv.imread(path)
			face,returnValue = detect_face_deep(img)
			if returnValue == 0:
				f.write(path)
				cv.imwrite('.\\'+ destination + '\\' + filename ,img)
			else:
				cv.imwrite('.\\'+ destination + '\\' + filename ,face)
				os.rename(path, '.\\' + backup + '\\' + filename)

	f.close()
	print(filecount , " files cropped in Time:" , time.time() - start_time)




def cropAllFaces(dir,destination,backup):
	start_time = time.time()
	directory = os.fsencode(dir)
	filecount = 0
	for file in os.listdir(directory):
	    filecount += 1
	    filename = os.fsdecode(file)
	    print(filename)
	    if filename.endswith(".png") or filename.endswith(".jpg"): 
	        path = ".\\" + dir + "\\" + filename
	        img = cv.imread(path)
	        face = detect_face(img)
	        if len(face) != 0:
	        	scaledface = cv.resize(face,(256,256))
	        else:
	        	scaledface = cv.resize(img,(256,256))
	        
	        cv.imwrite('.\\'+ destination + '\\' + filename ,scaledface)
	        os.rename(path, '.\\' + backup + '\\' + filename)

	print(filecount , " files cropped in Time:" , time.time() - start_time)


def labelAll():
	bins = [0.0, 6.0, 13.0, 20.0, 27.0, 34.0, 41.0, 48.0, 55.0, 62.0,np.inf]
	labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	with open('ground_truth/train_gt.csv', 'r') as f:
		reader = csv.reader(f)
		mydict = dict((rows[0],rows[1]) for rows in reader)
	labelSplit("train",mydict,bins,labels)

	with open('ground_truth/valid_gt.csv', 'r') as f:
		reader = csv.reader(f)
		mydict = dict((rows[0],rows[1]) for rows in reader)
	labelSplit("valid",mydict,bins,labels)
	with open('ground_truth/test_gt.csv', 'r') as f:
		reader = csv.reader(f)
		mydict = dict((rows[0],rows[1]) for rows in reader)
	labelSplit("test",mydict,bins,labels)


def labelSplit(dir,mydict,bins,labels):
	directory = os.fsencode(dir)
	
	filecount = 0
	for file in os.listdir(directory):
	    filecount += 1
	    filename = os.fsdecode(file)
	    print(filename)
	    if filename.endswith(".png") or filename.endswith(".jpg"): 
	        path = ".\\" + dir + "\\" + filename
	        where = arrangeLabel(mydict.get(filename),bins,labels)
	        shutil.copy2(path,'dataset_'+str(dir)+'\\'+str(where))
	        #os.rename(path, '.\\' + backup + '\\' + filename)

	#print(filecount , " files cropped in Time:" , time.time() - start_time)


def labelRanking():
	bins = [0.0, 6.0, 13.0, 20.0, 27.0, 34.0, 41.0, 48.0, 55.0, 62.0,np.inf]
	labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

	train_directory = "dataset_train"
	valid_directory = "dataset_valid"

	for label in labels:
		labelname = str(label)
		if not os.path.exists(labelname):
			os.makedirs(labelname,exist_ok=True)
			os.makedirs(labelname+"/0",exist_ok=True)
			os.makedirs(labelname+"/1",exist_ok=True)
			os.makedirs(labelname+"_valid",exist_ok=True)
			os.makedirs(labelname+"_valid/0",exist_ok=True)
			os.makedirs(labelname+"_valid/1",exist_ok=True)

		for dirName, subdirList, fileList in os.walk(train_directory):
			#print('Found directory: %s' % dirName)
			for fname in fileList:
				if fname.endswith(".png") or fname.endswith(".jpg"):
					path = dirName + "\\" + fname	
					print(fname)
					if (int(dirName[-1]) <= label):
						shutil.copy2(path,labelname+'\\'+str(0))
					else:
						shutil.copy2(path,labelname+'\\'+str(1))

		for dirName, subdirList, fileList in os.walk(valid_directory):
			#print('Found directory: %s' % dirName)
			for fname in fileList:
				if fname.endswith(".png") or fname.endswith(".jpg"):
					path = dirName + "\\" + fname	
					print(fname)
					if (int(dirName[-1]) <= label):
						shutil.copy2(path,labelname+"_valid\\"+str(0))
					else:
						shutil.copy2(path,labelname+"_valid\\"+str(1))




def arrangeLabel(age,bins,labels):
    for index in range(len(bins)):
        if float(age) > bins[index]:
            continue;
        else:
            return labels[index-1]
    return labels[index-1]


def resizeAll(dir):
	start_time = time.time()
	directory = os.fsencode(dir)
	filecount = 0
	for file in os.listdir(directory):
	    filecount += 1
	    filename = os.fsdecode(file)
	    print(filename)
	    if filename.endswith(".png") or filename.endswith(".jpg"): 
	        path = ".\\" + dir + "\\" + filename
	        img = cv.imread(path)
	        scaledface = cv.resize(img,(224,224))
	        
	        cv.imwrite('.\\'+ dir+ '\\' + filename ,scaledface)

	print(filecount ," files resized in Time:" , time.time() - start_time)


#cropAllWithDeep("valid","valid_cropped","valid_")

#cropAllFaces("test_2","testfaces_2_haar","test2")

#resizeAll("valid_deleted")

#labelAll()
labelRanking()
