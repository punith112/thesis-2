#IMPORTS
import sys
import os
import time
import uuid
import math

import numpy as np
import cv2
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('OpenCV Version: {}.{}.{}'.format(major_ver, minor_ver, subminor_ver))

import keras
from keras.models import load_model

#CONSTANTS
MODEL_PATH = os.path.join('model')
MODEL_FILE = os.path.join(MODEL_PATH, 'hand_model_gray.hdf5') # path to model weights and architechture file
MODEL_HISTORY = os.path.join(MODEL_PATH, 'model_history.txt') # path to model training history

#MAIN
hand_model = load_model(MODEL_FILE, compile = False)

classes = {
    0: 'forward',
    1: 'left',
    2: 'right',
    3: 'stop'
}

CURR_POSE = 'forward'
DATA = 'images'

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

# function for applying a mask to an array
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
}

# image count for file name
img_count = 0

cv2.namedWindow('mask')
cv2.createTrackbar('BL', 'mask', 0, 255, nothing)
cv2.createTrackbar('GL', 'mask', 0, 255, nothing)
cv2.createTrackbar('RL', 'mask', 150, 255, nothing)
cv2.createTrackbar('BU', 'mask', 255, 255, nothing)
cv2.createTrackbar('GU', 'mask', 255, 255, nothing)
cv2.createTrackbar('RU', 'mask', 255, 255, nothing)

while True:

    # read a new frame
    success, frame = video.read()
    if not success:
        # frame not successfully read from video capture
        break
    frame = cv2.flip(frame, 1)
    display = frame.copy()
    data_display = np.full_like(display, (51, 0, 51), dtype = np.uint8)

    blur = cv2.GaussianBlur(frame, (5, 5), 0) #blur
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV) #convert to the HSV color space

    #EXTRACTING THE HAND

    #special bars for skin segmentation
    BL = cv2.getTrackbarPos('BL', 'mask')
    GL = cv2.getTrackbarPos('GL', 'mask')
    RL = cv2.getTrackbarPos('RL', 'mask')
    BU = cv2.getTrackbarPos('BU', 'mask')
    GU = cv2.getTrackbarPos('GU', 'mask')
    RU = cv2.getTrackbarPos('RU', 'mask')

    lower = np.array([BL, GL, RL], dtype = "uint8")
    upper = np.array([BU, GU, RU], dtype = "uint8")

    skin_mask = cv2.inRange(hsv, lower, upper)

    # apply a series of erosions and dilations to the mask using an elliptical kernel
    #kernel = np.ones((5, 5), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations = 2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations = 2)

    # blur the mask to help remove noise, then apply the mask to the frame
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    imask = skin_mask > 0
    # get foreground from mask
    foreground = mask_array(frame, imask)

    #TRACKING

    # if tracking is active, update the tracker
    if tracking != -1:
        tracking, bbox = tracker.update(foreground)
        tracking = int(tracking)

    display_crop = display[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
    blur_crop = blur[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]
    hsv_crop = hsv[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]

    # draw bounding box
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    #cv2.rectangle(display, p1, p2, (51, 0, 51), 2, 1)

    # use numpy array indexing to crop the hand
    hand_crop = skin_mask[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]

    #GESTURE RECOGNITION

    if hand_crop.size != 0:
        # resize cropped hand and make prediction on gesture
        hand_crop_resized = np.expand_dims(cv2.resize(hand_crop, (54, 54)), axis=0).reshape((1, 54, 54, 1))
        prediction = hand_model.predict(hand_crop_resized)
        predi = prediction[0].argmax() # get the index of the greatest confidence
        gesture = classes[predi]

        for i, pred in enumerate(prediction[0]):
            # draw confidence bar for each gesture
            barx = positions['hand_pose'][0]
            bary = 60 + i*60
            bar_height = 20
            bar_length = int(400 * pred) + barx # calculate length of confidence bar

            colour = (25, 160, 255)

            cv2.putText(data_display, "{}: {}".format(classes[i], pred), (positions['hand_pose'][0], 30 + i*60), cv2.FONT_ITALIC , 0.75, (255, 255, 255), 2)
            cv2.rectangle(data_display, (barx, bary), (bar_length, bary - bar_height), colour, -1, 1)
        cv2.putText(display, "hand gesture: {}".format(gesture), positions['hand_pose'], cv2.FONT_ITALIC , 0.75, (25, 160, 255), 2)
    else:
        cv2.putText(display, "hand gesture: error", positions['hand_pose'], cv2.FONT_ITALIC , 0.75, (25, 160, 255), 2)

    # display result
    cv2.imshow("display", display)
    cv2.imshow("blur", blur)
    cv2.imshow("hsv", hsv)
    if hand_crop.size != 0:
        cv2.imshow('mask', hand_crop)
    cv2.imshow("data", data_display)

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
        fname = os.path.join('images', "stophand.jpg")
        cv2.imwrite(fname, display_crop)
        fname = os.path.join('images', "stophand_blur.jpg")
        cv2.imwrite(fname, blur_crop)
        fname = os.path.join('images', "stophand_hsv.jpg")
        cv2.imwrite(fname, hsv_crop)
    elif k != 255: print(k)

cv2.destroyAllWindows()
video.release()
