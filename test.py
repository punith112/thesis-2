import sys
import os
import time
import uuid
import math

import numpy as np
import cv2
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('OpenCV Version: {}.{}.{}'.format(major_ver, minor_ver, subminor_ver))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

img = cv2.imread('stophand.jpg')

blur = cv2.GaussianBlur(img, (5, 5), 0) #blur
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV) #convert to the HSV color space

h, s, v = cv2.split(hsv)
h = h.flatten()
s = s.flatten()
v = v.flatten()

#plotting
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(h, s, v)
plt.show()
