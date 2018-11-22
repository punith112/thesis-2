#IMPORTS
import sys # system functions (ie. exiting the program)
import os # operating system functions (ie. path building on Windows vs. MacOs)
import time # for time operations
import uuid # for generating unique file names
import math # math functions

import numpy as np # matrix operations (ie. difference between two matricies)
import cv2 # (OpenCV) computer vision functions (ie. tracking)
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('OpenCV Version: {}.{}.{}'.format(major_ver, minor_ver, subminor_ver))

import keras # high level api to tensorflow (or theano, CNTK, etc.) and useful image preprocessing
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
print('Keras image data format: {}'.format(K.image_data_format()))

from sklearn.cluster import KMeans
from sklearn import metrics
from collections import Counter

#CONSTANTS
IMAGES_FOLDER = os.path.join('images') # images for visuals

MODEL_PATH = os.path.join('model')
MODEL_FILE = os.path.join(MODEL_PATH, 'hand_model_gray.hdf5') # path to model weights and architechture file
MODEL_HISTORY = os.path.join(MODEL_PATH, 'model_history.txt') # path to model training history

#MAIN

def dominant_color(image):
    image_array = image.reshape((image.shape[0] * image.shape[1], 3))
    # Clusters the pixels
    clt = KMeans(n_clusters = 3)
    labels = clt.fit_predict(image_array)
    #count labels to find most popular
    label_counts = Counter(labels)
    #subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common(1)[0][0]]
    return list(dominant_color)

def find_fist_color(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    dcolor = 0
    roi_color = 0
    fists = fist_cascade.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (60,60))
    if len(fists) != 0:
        maxFist = (0,0,0,0)
        for fist in fists:
            if fist[2] > maxFist[2]:
                maxFist = fist;
        x,y,w,h = maxFist
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
        dcolor = dominant_color(roi_color)
    return dcolor

fist_cascade = cv2.CascadeClassifier('aGest.xml')

def nothing(x): #needed for createTrackbar to work in python.
    pass

#lower = np.array([0, 50, 0], dtype = "uint8")
#upper = np.array([120, 150, 255], dtype = "uint8")

video = cv2.VideoCapture(0)
color_detected = False
dcolor = [0, 0, 0]

cv2.namedWindow('HSV settings')

while True:
    #time.sleep(0.025)

    timer = cv2.getTickCount()

    # Read a new frame
    success, frame = video.read()
    if not success:
        # Frame not successfully read from video capture
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)

    #EXTRACTING THE HAND

    #skin color detection
    if color_detected == False:
        dcolor = find_fist_color(frame)
        if dcolor != 0:
            color_detected = True

            BL = int(round((255 + dcolor[0] - 50) % 255))
            GL = int(round((255 + dcolor[0] - 20) % 255))
            RL = int(round((255 + dcolor[0] - 20) % 255))

            BU = int(round((255 + dcolor[0] + 50) % 255))
            GU = int(round((255 + dcolor[0] + 20) % 255))
            RU = int(round((255 + dcolor[0] + 20) % 255))

            cv2.destroyWindow('HSV settings')
            cv2.namedWindow('HSV settings')
            cv2.createTrackbar('BL', 'HSV settings', BL, 255, nothing)
            cv2.createTrackbar('GL', 'HSV settings', GL, 255, nothing)
            cv2.createTrackbar('RL', 'HSV settings', RL, 255, nothing)
            cv2.createTrackbar('BU', 'HSV settings', BU, 255, nothing)
            cv2.createTrackbar('GU', 'HSV settings', GU, 255, nothing)
            cv2.createTrackbar('RU', 'HSV settings', RU, 255, nothing)

    #precalculated values of skin color
    #special bars for skin segmentation
    #frame difference method

    if dcolor != 0:

        lower = np.array([BL, GL, RL], dtype = "uint8")
        upper = np.array([BU, GU, RU], dtype = "uint8")

        BL = cv2.getTrackbarPos('BL', 'HSV settings')
        GL = cv2.getTrackbarPos('GL', 'HSV settings')
        RL = cv2.getTrackbarPos('RL', 'HSV settings')
        BU = cv2.getTrackbarPos('BU', 'HSV settings')
        GU = cv2.getTrackbarPos('GU', 'HSV settings')
        RU = cv2.getTrackbarPos('RU', 'HSV settings')


        #lower = np.array([(255 + dcolor[0]) % 255 - 50, (255 + dcolor[0]) % 255 - 20, (255 + dcolor[0]) % 255 - 20], dtype = "uint8")
        #upper = np.array([(255 + dcolor[0]) % 255 + 50, (255 + dcolor[0]) % 255 + 20, (255 + dcolor[0]) % 255 + 20], dtype = "uint8")

        skin_mask = cv2.inRange(frame, lower, upper)

    # apply a series of erosions and dilations to the mask
	# using an elliptical kernel
    #kernel = np.ones((5, 5), np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.erode(skin_mask, kernel, iterations = 2)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations = 2)

    # blur the mask to help remove noise, then apply the
	# mask to the frame
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

        cv2.imshow('HSV settings', skin_mask)

    #fgmask = fgbg.apply(frame)

    # Apply erosion to clean up noise
    #if ERODE:
    #    fgmask = cv2.erode(fgmask, np.ones((3,3), dtype=np.uint8), iterations=1)

    # Calculate Frames per second (FPS)
    #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    # Display FPS on frame
    #cv2.putText(frame, "FPS : " + str(int(fps)), (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Display result
    cv2.imshow("frame", frame)
    # cv2.imshow("fgmask", fgmask)

    k = cv2.waitKey(1) & 0xff
    if k == 27:# escape pressed
        break
    elif k == 115: # s pressed
        fname = input("File name")
        cv2.imwrite(os.path.join(IMAGES_FOLDER, '{}.jpg'.format(fname)), frame)
    elif k == 100:
        color_detected = False
        dcolor = 0
        print('d key pressed')

cv2.destroyAllWindows()
video.release()
