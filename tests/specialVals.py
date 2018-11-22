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
classes = {
    0: 'stop',
    1: 'forward',
    2: 'right',
    3: 'left'
}

CURR_POSE = 'forward'
DATA = 'validation_data'

#hand_model = load_model(MODEL_FILE, compile=False)

# set up tracker
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

# helper function for applying a mask to an array
def mask_array(array, imask):
    if array.shape[:2] != imask.shape:
        raise Exception("Shapes of input and imask are incompatible")
    output = np.zeros_like(array, dtype=np.uint8)
    for i, row in enumerate(imask):
        output[i, row] = array[i, row]
    return output

def nothing(x): #needed for createTrackbar to work
    pass

video = cv2.VideoCapture(0)

# read first frame
success, frame = video.read()
if not success:
    print("first frame not read")
    sys.exit()
frame = cv2.flip(frame, 1)

# bounding box -> (TopRightX, TopRightY, Width, Height)
bbox_initial = (60, 60, 200, 200)
bbox = bbox_initial

# Select roi for bbox
#bbox = cv2.selectROI(frame, False)
#cv2.destroyAllWindows()

# tracking status, -1 for not tracking, 0 for unsuccessful tracking, 1 for successful tracking
tracking = -1

# text display positions
positions = {
    'hand_pose': (15, 40), # hand pose text
    'fps': (15, 20), # fps counter
    'null_pos': (200, 200) # used as null point for mouse control
}

# image count for file name
img_count = 0

while True:
    # read a new frame
    success, frame = video.read()
    if not success:
        # frame not successfully read from video capture
        break
    frame = cv2.flip(frame, 1)
    display = frame.copy()

    # Our operations on the frame come here
    # resize the frame, convert it to the HSV color space,
	# and determine the HSV pixel intensities that fall into
	# the speicifed upper and lower boundaries
    blur = cv2.GaussianBlur(frame, (3, 3), 0) #blur
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    ycbcr = cv2.cvtColor(blur, cv2.COLOR_BGR2YCR_CB)

    #EXTRACTING THE HAND

    #skin color detection
    #precalculated values of skin color


    # get foreground from mask
    #foreground = mask_array(frame, imask)
    #foreground_display = foreground.copy()

    #TRACKING

    # if tracking is active, update the tracker
    if tracking != -1:
        tracking, bbox = tracker.update(foreground)
        tracking = int(tracking)

    # draw bounding box
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(display, p1, p2, (255, 0, 0), 2, 1)

    # use numpy array indexing to crop the hand
    hand_crop = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]

    def checkRGB(px):
        R, G, B = px
        return ( ((R > 95) and (G > 40) and (B > 20))
            and ((max(px)-min(px))>15) and (abs(R - G) > 15) and
            (R > G) and (R > B))

    def checkRGB2(px):
        B,G,R = px
        return ((R >220) and (G > 210) and (B > 170) and
            (abs(R - G) <= 15) and (R > B) and (G > B))

    def checkCbCr(px):
        Y, Cb, Cr = px
        return (Cr <= 1.5862 * Cb + 20) and (Cr >= 0.3448 * Cb + 76.2069) and (Cr >= -4.5652 * Cb + 234.565) and (Cr <= -1.5 * Cb + 301.75) and (Cr <= -2.2857 * Cb + 432.85)

    def checkHSV(px):
        H, S, V = px
        return (H < 25) or (H > 230)

    def iterate_over_list(img):  # your method
        img = img.tolist()
        skinmask =  [[(1 if ((checkRGB(px) or checkRGB2(px)) and checkCbCr(px) and checkHSV(px)) else 0) for px in row] for row in img]
        return skinmask

    skin_mask = iterate_over_list(hand_crop)
    mask = np.array(skin_mask, dtype = "uint8")
    skin_mask = cv2.bitwise_and(hand_crop, hand_crop, mask = mask)

    #frame difference method

    # apply a series of erosions and dilations to the mask
	# using an elliptical kernel
    #kernel = np.ones((5, 5), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations = 2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations = 2)

    # blur the mask to help remove noise, then apply the
	# mask to the frame
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    imask = skin_mask > 0

    '''
    try:
        # Resize cropped hand and make prediction on gesture
        hand_crop_resized = np.expand_dims(cv2.resize(hand_crop, (54, 54)), axis=0).reshape((1, 54, 54, 1))
        prediction = hand_model.predict(hand_crop_resized)
        predi = prediction[0].argmax() # Get the index of the greatest confidence
        gesture = classes[predi]

        for i, pred in enumerate(prediction[0]):
            # Draw confidence bar for each gesture
            barx = positions['hand_pose'][0]
            bary = 60 + i*60
            bar_height = 20
            bar_length = int(400 * pred) + barx # calculate length of confidence bar

            # Make the most confidence prediction green
            if i == predi:
                colour = (0, 255, 0)
            else:
                colour = (0, 0, 255)

            cv2.putText(data_display, "{}: {}".format(classes[i], pred), (positions['hand_pose'][0], 30 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.rectangle(data_display, (barx, bary), (bar_length, bary - bar_height), colour, -1, 1)

        cv2.putText(display, "hand pose: {}".format(gesture), positions['hand_pose'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(foreground_display, "hand pose: {}".format(gesture), positions['hand_pose'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    except Exception as ex:
        cv2.putText(display, "hand pose: error", positions['hand_pose'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        cv2.putText(foreground_display, "hand pose: error", positions['hand_pose'], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    '''

    # display result
    cv2.imshow("display", display)
    if hand_crop.size != 0:
        cv2.imshow('mask', skin_mask)
    #cv2.imshow("data", data_display)

    k = cv2.waitKey(1) & 0xff

    if k == 27:# escape pressed
        break
    elif k == 114 or k == 112:
        # r pressed
        tracking = -1
        bbox = bbox_initial
    elif k == 116:
        # t pressed
        # initialize tracker with first frame and bounding box
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
