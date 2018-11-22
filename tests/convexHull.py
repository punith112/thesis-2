import sys # system functions (ie. exiting the program)
import os # operating system functions (ie. path building on Windows vs. MacOs)
import time # for time operations
import uuid # for generating unique file names
import math # math functions
import imutils

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

classes = {
    0: 'stop',
    1: 'forward',
    2: 'right',
    3: 'left'
}

CURR_POSE = 'forward'
DATA = 'validation_data'

# Set up tracker.
def setup_tracker(ttype):
    tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[ttype]

    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()

    return tracker

# Helper function for applying a mask to an array
def mask_array(array, imask):
    if array.shape[:2] != imask.shape:
        raise Exception("Shapes of input and imask are incompatible")
    output = np.zeros_like(array, dtype=np.uint8)
    for i, row in enumerate(imask):
        output[i, row] = array[i, row]
    return output

def nothing(x): #needed for createTrackbar to work in python.
    pass

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 50, 0], dtype = "uint8")
upper = np.array([120, 150, 255], dtype = "uint8")

video = cv2.VideoCapture(0)

# Read first frame
success, frame = video.read()
if not success:
    print("first frame not read")
    sys.exit()
frame = cv2.flip(frame, 1)

# Tracking
# Bounding box -> (TopRightX, TopRightY, Width, Height)
bbox_initial = (100, 100, 100, 100)
bbox = bbox_initial
# Select roi for bbox
#bbox = cv2.selectROI(frame, False)
#cv2.destroyAllWindows()
# Tracking status, -1 for not tracking, 0 for unsuccessful tracking, 1 for successful tracking
tracking = -1

# Text display positions
positions = {
    'hand_pose': (15, 40),
    'fps': (15, 20)
}

# Image count for file name
img_count = 0

cv2.namedWindow('HSV settings')
cv2.createTrackbar('BL', 'HSV settings', 0, 255, nothing)
cv2.createTrackbar('GL', 'HSV settings', 0, 255, nothing)
cv2.createTrackbar('RL', 'HSV settings', 150, 255, nothing)
cv2.createTrackbar('BU', 'HSV settings', 255, 255, nothing)
cv2.createTrackbar('GU', 'HSV settings', 255, 255, nothing)
cv2.createTrackbar('RU', 'HSV settings', 255, 255, nothing)

while True:
    #time.sleep(0.025)

    timer = cv2.getTickCount()

    # Read a new frame
    success, frame = video.read()
    if not success:
        # Frame not successfully read from video capture
        break
    frame = cv2.flip(frame, 1)
    display = frame.copy()

    # Our operations on the frame come here
    # resize the frame, convert it to the HSV color space,
	# and determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
    #frame = imutils.resize(frame, width = 400)
    blur = cv2.GaussianBlur(frame, (3, 3), 0) #blur
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    #EXTRACTING THE HAND

    #skin color detection
    #precalculated values of skin color

    #special bars for skin segmentation
    BL = cv2.getTrackbarPos('BL', 'HSV settings')
    GL = cv2.getTrackbarPos('GL', 'HSV settings')
    RL = cv2.getTrackbarPos('RL', 'HSV settings')
    BU = cv2.getTrackbarPos('BU', 'HSV settings')
    GU = cv2.getTrackbarPos('GU', 'HSV settings')
    RU = cv2.getTrackbarPos('RU', 'HSV settings')

    lower = np.array([BL, GL, RL], dtype = "uint8")
    upper = np.array([BU, GU, RU], dtype = "uint8")

    skin_mask = cv2.inRange(hsv, lower, upper)

    #frame difference method

    # apply a series of erosions and dilations to the mask
	# using an elliptical kernel
    #kernel = np.ones((5, 5), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    erosion = cv2.erode(skin_mask, kernel, iterations = 2)
    dilation = cv2.dilate(skin_mask, kernel, iterations = 2)
    opening = cv2.erode(skin_mask, kernel, iterations = 2)
    opening = cv2.dilate(opening, kernel, iterations = 2)

    '''
    # blur the mask to help remove noise, then apply the
	# mask to the frame
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    imask = skin_mask > 0
    # Get foreground from mask
    foreground = mask_array(frame, imask)
    foreground_display = foreground.copy()
    #ret, skin_mask = cv2.threshold(skin_mask, 127, 255, 0)
    #skin = cv2.bitwise_and(frame, frame, mask = skin_mask)

    # If tracking is active, update the tracker
    if tracking != -1:
        tracking, bbox = tracker.update(foreground)
        tracking = int(tracking)

    # Draw bounding box
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(foreground_display, p1, p2, (255, 0, 0), 2, 1)
    cv2.rectangle(display, p1, p2, (255, 0, 0), 2, 1)

    # Use numpy array indexing to crop the hand
    mask_crop = skin_mask[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
    hand_crop = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]

    # find contours in thresholded image, then grab the largest
    # one
    cnts = cv2.findContours(mask_crop.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    try:
        cnt = max(cnts, key = cv2.contourArea)
        cv2.drawContours(hand_crop, [cnt], 0, (0, 255, 255), 2)

        # convex hull
        hull = cv2.convexHull(cnt)
        cv2.drawContours(hand_crop, [hull], -1, (255, 0, 0), 2)

        # center of the contours
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(hand_crop, (cX, cY), 5, (0, 0, 0), -1)

        # convexity defects
        hull = cv2.convexHull(cnt, returnPoints = False)
        defects = cv2.convexityDefects(cnt, hull)

        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            #cv2.circle(crop_image, far, 5, [0, 255, 0], -1)

            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14

            # if angle > 90 draw a circle at the far point
            if angle <= 90:
                count_defects += 1
                cv2.circle(hand_crop, far, 5, [0, 255, 0], -1)
    except:
        pass
'''

    # Display result
    cv2.imshow("display", display)
    cv2.imshow("mask", skin_mask)
    cv2.imshow("erosion", erosion)
    cv2.imshow("dilation", dilation)
    cv2.imshow("opening", opening)
    '''
    if mask_crop.size != 0:
        cv2.imshow('HSV settings', mask_crop)
        cv2.imshow('hand', hand_crop)
    '''
    # cv2.imshow("fgmask", fgmask)

    k = cv2.waitKey(1) & 0xff

    if k == 27:# escape pressed
        break
    elif k == 114 or k == 112:
        # r pressed
        tracking = -1
        bbox = bbox_initial
    elif k == 116:
        # t pressed
        # Initialize tracker with first frame and bounding box
        tracker = setup_tracker(2)
        tracking = tracker.init(frame, bbox)
    elif k == 115:
        # s pressed
        img_count += 1
        fname = os.path.join(DATA, CURR_POSE, "{}_{}.jpg".format(CURR_POSE, img_count))
        cv2.imwrite(fname, hand_crop)
    elif k != 255: print(k)

cv2.destroyAllWindows()
video.release()
