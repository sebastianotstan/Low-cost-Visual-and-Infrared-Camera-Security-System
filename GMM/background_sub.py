# import cv2
# import numpy as np


# # print (cv2.__version__)
# cap = cv2.VideoCapture(0)
# fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
# fgbgAdaptiveGaussain = cv2.createBackgroundSubtractorMOG2()
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# fgbgBayesianSegmentation = cv2.bgsegm.createBackgroundSubtractorGMG()

# while(1):
#   ret, frame = cap.read()
#   fgmask = fgbg.apply(frame)
#   fgbgAdaptiveGaussainmask = fgbgAdaptiveGaussain.apply(frame)
#   fgbgBayesianSegmentationmask = fgbgBayesianSegmentation.apply(frame)
#   fgbgBayesianSegmentationmask = cv2.morphologyEx(fgbgBayesianSegmentationmask,cv2.MORPH_OPEN,kernel)
  
#   cv2.namedWindow('Background Subtraction Bayesian Segmentation',0)
#   cv2.namedWindow('Background Subtraction',0)
#   cv2.namedWindow('Background Subtraction Adaptive Gaussian',0)
#   cv2.namedWindow('Original',0)

#   cv2.resizeWindow('Original', 300,300)
#   cv2.imshow('Background Subtraction Bayesian Segmentation',fgbgBayesianSegmentationmask)
#   cv2.imshow('Background Subtraction',fgmask)
#   cv2.imshow('Background Subtraction Adaptive Gaussian',fgbgAdaptiveGaussainmask)
#   cv2.imshow('Original',frame)
  
#   k = cv2.waitKey(1) & 0xff
  
#   if k==ord('q'):
#     break

# cap.release()
# cv2.destroyAllWindows()
# print ('Program Closed')

import numpy as np
import cv2
# Specify folders
filepath = '/Users/sebbyma/Desktop/Imperial_Fourth_Year/FYP/code/GMM'
# Get a video in the folder
video = cv2.VideoCapture(filepath + '/insulation.mp4')
# Custom convolution kernel -- rectangular , For morphological processing
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# Create Gaussian mixture model for background modeling
back = cv2.createBackgroundSubtractorMOG2()
# Read and process each frame
count = 0
sum = 0
alert = False
while True:
    ret, frame = video.read() # Read one frame at a time , Returns whether to open and each frame of the image
    sum = sum + 1
    img = back.apply(frame) # Background modeling
    # Open operation （ Corrosion before expansion ）, Noise removal
    img_close = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # Contour detection , Get the outermost contour , Keep only the end coordinates
    contours,hierarchy = cv2.findContours(img_close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # Calculate the outline circumscribed rectangle
    for cnt in contours:
        # Calculate contour perimeter
        length = cv2.arcLength(cnt,True)
        if length>5000:
            # Get the elements of the circumscribed rectangle

            x,y,w,h = cv2.boundingRect(cnt)
            # Draw this rectangle , Draw... On the original video frame image , Top left coordinates (x,y), Lower right coordinates (x+w,y+h)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
            alert = True
    # Image display
    cv2.imshow('frame',frame) # Original picture
    cv2.imshow('img',img) # Gaussian model diagram
    if alert:
        count = count + 1

    if sum == 20:
        if count > 10:
            conf = count/sum * 100
            print('There is', conf, '% chance of a human under an insulation blanket nearby')
        count = 0
        sum = 0

    alert = False
    # Set closing conditions , One frame 200 millisecond
    k = cv2.waitKey(100) & 0xff
    if k == 27: #27 Represents the exit key ESC
        break
# Release resources
video.release()
cv2.destroyAllWindows()