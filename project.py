import numpy as np
import cv2
from keras.models import load_model
from keras_vggface import utils
from keras_vggface.vggface import VGGFace
from keras import backend as K
import os
import time
from keras.preprocessing.image import img_to_array
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import csv
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import threading

class RankingCNN:
    def __init__(self):
        print("[INFO] loading face detection model...")
        self.net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

        print("[INFO] loading networks...")
        self.models = []
        for i in range(9):
            print("[INFO] loading Ranking"+str(i)+"...")
            self.models.append(load_model("Ranking"+str(i)))
            
        self.ff = 0
        self.ages = [-1] * 5
        self.bins = ["0-6", "7-13", "14-20", "21-27", "28-34", "35-41", "42-48", "49-55", "56-62", "+63"]

    def webcam(self):
        self.cap = cv2.VideoCapture(0)
        self.frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while(True):
            self.ff += 1
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            self.ageEstimate(frame)
            cv2.putText(frame, "Press q to exit", (0, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow('Ranking-CNN Age Estimation',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.ff = 0
        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()

    def image(self):
        filetypes = (("Image Files","*.jpg;*.png;*.jpeg"),("all files","*.*"))
        filename =  filedialog.askopenfilename(initialdir=os.getcwd(),title = "Select a image",filetypes = filetypes)
        if filename != None:
            img = cv2.imread(filename)
            self.ageEstimate(img)
            cv2.imshow('Ranking-CNN Age Estimation',img)
            

    def test_window(self):
        window = tk.Toplevel(root)
        progress = Progressbar(window, orient=HORIZONTAL,length=1978,  mode='indeterminate')
        progress.pack()
        progress['value']=0
        

    def test(self):
        with open('test_gt.csv', 'r') as f:
            reader = csv.reader(f)
            mydict = dict((rows[0],rows[1]) for rows in reader)

        x_test,y_test = self.readFaces("test_faces",mydict)
        bins = [0.0, 6.0, 13.0, 20.0, 27.0, 34.0, 41.0, 48.0, 55.0, 62.0,np.inf]
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        y_test = np.array(self.arrangeLabels(y_test,bins,labels))
        predictions = []
        hit = 0
        total = len(x_test)
        for index in range(total):
            # classify the input image
            agg = 0
            for model in self.models:  
                proba = model.predict(x_test[index])[0]
                idx = np.argmax(proba)
                if(idx == 1):
                    agg += 1
            predictions.append(agg)
            hit += agg==y_test[index]

        cm = confusion_matrix(y_test,predictions)
        print(hit/total)
        newwin = tk.Toplevel(root)
        display = tk.Label(newwin, text=str(hit/total))
        display.pack()
        cm = pd.DataFrame(cm)
        sns.heatmap(cm, annot=True, fmt = "d",cmap="YlGnBu")
        plt.show()

    def ageEstimate(self,frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        # Display the resulting frame
        # loop over the detections

        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
         
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > 0.7:
                # compute the (x, y)-coordinates of the bounding box for the
                # object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                if self.ff%10 == 0 and startX > 0 and startY > 0 and endX < w and endY <h:
                    sub_face = frame[startY:endY, startX:endX]
                    img = cv2.resize(sub_face, (224, 224))
                    img = img.astype("float16")
                    img = utils.preprocess_input(img)
                    img = img_to_array(img)
                    img = np.expand_dims(img, axis=0)
                    agg = 0
                    for model in self.models:  
                        proba = model.predict(img)[0]
                        idx = np.argmax(proba)
                        if(idx == 1):
                            agg += 1
                    self.ages[i] = agg
                # draw the bounding box of the face along with the associated
                # probability

                y = startY - 10 if startY - 10 > 10 else startY + 10
                if startX > 0 and startY > 0 and endX < w and endY <h:
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                        (0, 0, 255), 2)
                    cv2.putText(frame, str(self.bins[self.ages[i]]), (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def readFaces(self,dir,mydict):
        faces = []
        ages = []
        directory = os.fsencode(dir)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".png") or filename.endswith(".jpg"): 
                path = ".\\" + dir + "\\" + filename
                img = cv2.imread(path)
                img = cv2.resize(img, (224, 224))
                img = img.astype("float16")
                img = img_to_array(img)
                img = utils.preprocess_input(x=img,version=1)
                img = np.expand_dims(img, axis=0)
                faces.append(img)
                ages.append(mydict.get(filename))
        return np.asarray(faces),ages

    def arrangeLabels(self,Y,bins,labels):
        returnLabels = []
        for label in Y:
            for index in range(len(bins)):
                if float(label) > bins[index]:
                    continue;
                else:
                    returnLabels.append(labels[index-1])
                    break;
                    
        return returnLabels


rankingCNN = RankingCNN()

root = tk.Tk()
root.title("Age Estimation Tool")

img = ImageTk.PhotoImage(Image.open("logo.PNG"))
panel = tk.Label(root, image = img)
panel.pack(side = "top", fill = "both", expand = "no")


frame = tk.Frame(root)
frame.pack()

quit = tk.Button(root,
                   text="QUIT", 
                   fg="red",
                   command=quit)
quit.pack(side=tk.RIGHT)


webcam = tk.Button(root,
                   text="Webcam",
                   command=rankingCNN.webcam)
webcam.pack(side=tk.RIGHT)

image = tk.Button(root,
                   text="Choose Image",
                   command=rankingCNN.image)
image.pack(side=tk.RIGHT)

test = tk.Button(root,
                   text="rankingCNN",
                   command=rankingCNN.test)
test.pack(side=tk.RIGHT)


root.mainloop()