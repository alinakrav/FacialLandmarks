# This combines the facial alignment class and the facial landmark detection class from pyimagesearch.com's API

# import the necessary packages
from imutils import face_utils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import numpy as np
import argparse
import imutils
import dlib
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--shape-predictor", required=False,
# 	help="path to facial landmark predictor")
ap.add_argument("--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
############## i added this for alignment
# initialize the face aligner
fa = FaceAligner(predictor, (0.35,0.35), 350)
#########

# load the input image, resize it, and convert it to grayscale
image = cv2.imread("images/" + args["image"])
image = imutils.resize(image, width=500)

# cv2.imshow("original", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the grayscale image
rects = detector(gray, 1)

# loop over the face detections
for (i, rect) in enumerate(rects):
	######### added this
	faceAligned = fa.align(image, gray, rect)
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	# shape = predictor(gray, rect)
	wholeBox = (dlib.rectangle(left=0, top=0, right=faceAligned.shape[0], bottom=faceAligned.shape[1]))
	shape = predictor(faceAligned, wholeBox)
	shape = face_utils.shape_to_np(shape)

	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	# (x, y, w, h) = face_utils.rect_to_bb(rect)



	# cv2.imshow("Aligned", faceAligned)
	# cv2.waitKey(0)
	#############
	# cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# show the face number
	# cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
	# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x, y)-coordinates for the facial landmarks
	# and draw them on the image
	cv2.rectangle(faceAligned, (0,0), (faceAligned.shape[0], faceAligned.shape[1]), (255,255,255),-1)
	for (x, y) in shape:
		cv2.circle(faceAligned, (x, y), 2, (0, 0, 255), -1)
#
# # show the output image with the face detections + facial landmarks
cv2.imshow("original", image)
cv2.imshow("Output", faceAligned)
cv2.waitKey(0)
