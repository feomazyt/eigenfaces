# USAGE
# python eigenfaces.py --input caltech_faces --visualize 1

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.exposure import rescale_intensity
from pyimagesearch.faces import load_face_dataset
from imutils import build_montages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
from tkinter import *
from PIL import Image, ImageTk


win = Tk()
win.geometry("512x288")
label = Label(win)
label.grid(row=0, column=0)
FACE_GRAY = None
FIN = None
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')


# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector",
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the CALTECH faces dataset
print("[INFO] loading dataset...")
(faces_, labels_) = load_face_dataset("caltech_faces", net,
	minConfidence=0.5, minSamples=20)
print("[INFO] {} images in dataset".format(len(faces_)))

def train_model(faces, labels):
	# flatten all 2D faces into a 1D list of pixel intensities
	pcaFaces = np.array([f.flatten() for f in faces])


	# encode the string labels as integers
	le = LabelEncoder()
	labels = le.fit_transform(labels)

	# construct our training and testing split
	split = train_test_split(faces, pcaFaces, labels, test_size=0.25,
		stratify=labels, random_state=42)
	(origTrain, origTest, trainX, testX, trainY, testY) = split

	# compute the PCA (eigenfaces) representation of the data, then
	# project the training data onto the eigenfaces subspace
	print("[INFO] creating eigenfaces...")
	pca = PCA(
		svd_solver="randomized",
		n_components=150,
		whiten=True)
	start = time.time()
	trainX = pca.fit_transform(trainX)
	end = time.time()
	print("[INFO] computing eigenfaces took {:.4f} seconds".format(
		end - start))

	# train a classifier on the eigenfaces representation
	print("[INFO] training classifier...")
	model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42)
	model.fit(trainX, trainY)
	return model, pca, le

name = ""
model_ = None
pca_ = None
le_ = None

while True:
	ret, frame = cap.read()
	width = int(frame.shape[1] * 40 / 100)
	height = int(frame.shape[0] * 40 / 100)
	dim = (width, height)
	frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	face = face_cascade.detectMultiScale(gray, 1.2, 6)
	for (x, y, w, h) in face:
		# cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
		FACE_GRAY = gray[y:y + w, x:x + w]
		# fin = edges(face_gray)
		try:
			FIN = cv2.resize(FIN.astype(np.uint8), (150, 150), interpolation=cv2.INTER_AREA)
			cv2.imshow("nms", FIN)
		except:
			pass

		face_color = frame[y:y + h, x:x + w]
		# edges = cv2.Canny(frame, 60, 100)

		eyes = eye_cascade.detectMultiScale(FACE_GRAY, 1.2, 4)
		smiles = smile_cascade.detectMultiScale(FACE_GRAY, 1.3, 7)
		# ret, thresh1 = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY)

		for (ex, ey, ew, eh) in eyes:
			cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
		for (sx, sy, sw, sh) in smiles:
			cv2.rectangle(face_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
		if(model_):
			test = np.array([cv2.resize(FACE_GRAY, (47, 62)).flatten()]) #.reshape(1, -1)
			predictions = model_.predict(pca_.transform(test))
			predName = le_.inverse_transform([predictions[0]])[0]
			cv2.putText(frame, predName, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

	cv2.imshow('frame', frame)

	if cv2.waitKey(1) == ord('q'):
		break
	if cv2.waitKey(1) == ord('l'):
		name = input("podaj imie: ")
		print(name)
	if cv2.waitKey(1) == ord('c'):
		if name != "":
			# faceROI = frame[face[0,0]:face[0,1], face[0,2]:face[0,3]]
			faceROI = cv2.resize(FACE_GRAY, (47, 62))
			# faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
			faces_ = list(faces_)
			faces_.append(faceROI)
			faces_ = np.array(faces_)

			labels_ = list(labels_)
			labels_.append(name)
			labels_ = np.array(labels_)
			print("samle get")
		else:
			print("enter label first")
	if cv2.waitKey(1) == ord('t'):
		print("train placeholder")
		model_, pca_, le_  = train_model(faces_, labels_)
		name = ""

	# fin = edges(face_gray)
	# cv2.imshow("nms", fin.astype(np.uint8))

# cv2.waitKey()
cap.release()
cv2.destroyAllWindows()