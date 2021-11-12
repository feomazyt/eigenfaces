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
(faces, labels) = load_face_dataset("caltech_faces", net,
	minConfidence=0.5, minSamples=20)
print("[INFO] {} images in dataset".format(len(faces)))


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

# check to see if the PCA components should be visualized
if 0 > 0:
	# initialize the list of images in the montage
	images = []

	# loop over the first 16 individual components
	for (i, component) in enumerate(pca.components_[:16]):
		# reshape the component to a 2D matrix, then convert the data
		# type to an unsigned 8-bit integer so it can be displayed
		# with OpenCV
		component = component.reshape((62, 47))
		component = rescale_intensity(component, out_range=(0, 255))
		component = np.dstack([component.astype("uint8")] * 3)
		images.append(component)

	# construct the montage for the images
	montage = build_montages(images, (47, 62), (4, 4))[0]

	# show the mean and principal component visualizations
	# show the mean image
	mean = pca.mean_.reshape((62, 47))
	mean = rescale_intensity(mean, out_range=(0, 255)).astype("uint8")
	cv2.imshow("Mean", mean)
	cv2.imshow("Components", montage)
	cv2.waitKey(0)

# train a classifier on the eigenfaces representation
print("[INFO] training classifier...")
model = SVC(kernel="rbf", C=10.0, gamma=0.001, random_state=42)
model.fit(trainX, trainY)

# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(pca.transform(testX))
print(classification_report(testY, predictions,
	target_names=le.classes_))

# generate a sample of testing data
idxs = np.random.choice(range(0, len(testY)), size=10, replace=False)

# loop over a sample of the testing data
for i in idxs:
	# grab the predicted name and actual name
	predName = le.inverse_transform([predictions[i]])[0]
	actualName = le.classes_[testY[i]]

	# grab the face image and resize it such that we can easily see
	# it on our screen
	face = np.dstack([origTest[i]] * 3)
	face = imutils.resize(face, width=250)

	# draw the predicted name and actual name on the image
	cv2.putText(face, "pred: {}".format(predName), (5, 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
	cv2.putText(face, "actual: {}".format(actualName), (5, 60),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

	# display the predicted name  and actual name
	print("[INFO] prediction: {}, actual: {}".format(
		predName, actualName))

	# display the current face to our screen
	cv2.imshow("Face", face)
	cv2.waitKey(0)

	while True:
		ret, frame = cap.read()
		width = int(frame.shape[1] * 70 / 100)
		height = int(frame.shape[0] * 70 / 100)
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

		cv2.imshow('frame', frame)

		if cv2.waitKey(1) == ord('q'):
			break
		if cv2.waitKey(1) == ord('w'):
			print("test")
		# fin = edges(face_gray)
		# cv2.imshow("nms", fin.astype(np.uint8))

	# cv2.waitKey()
	cap.release()
	cv2.destroyAllWindows()