# -*- coding: utf-8 -*-
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
# ---------------------------
# --image : The path to the input image. We will attempt to recognize the faces in this image.
# --detector : The path to OpenCV’s deep learning face detector. We’ll use this model to detect where in the image the face ROIs are.
# --embedding-model : The path to OpenCV’s deep learning face embedding model. We’ll use this model to extract the 128-D face embedding from the face ROI — we’ll feed the data into the recognizer.
# --recognizer : The path to our recognizer model. We trained our SVM recognizer in Step #2. This is what will actually determine who a face is.
# --le : The path to our label encoder. This contains our face labels such as 'adrian'  or 'trisha' .
# --confidence : The optional threshold to filter weak face detections.

import json
ap = argparse.ArgumentParser()
ap.add_argument("-cfg", "--config", default = "recognize.json", help = "Path to the config file (default one is extract_embeddings.json)")
args = vars(ap.parse_args())

cfg = json.loads(open(args["config"]).read())
# ---------------------------

# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([cfg["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([cfg["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
 
# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(cfg["embedding_model"])
 
# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(cfg["recognizer"], "rb").read())
le = pickle.loads(open(cfg["le"], "rb").read())

image = cv2.imread(cfg["image"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]
 
# construct a blob from the image
imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
 
# apply OpenCV's deep learning-based face detector to localize
# faces in the input image
detector.setInput(imageBlob)
detections = detector.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the
    # prediction
    confidence = detections[0, 0, i, 2]

    # filter out weak detections
    if confidence > cfg["confidence"]:
        # compute the (x, y)-coordinates of the bounding box for the
        # face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # extract the face ROI
        face = image[startY:endY, startX:endX]
        (fH, fW) = face.shape[:2]

        # ensure the face width and height are sufficiently large
        if fW < 20 or fH < 20:
            continue

        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        embedder.setInput(faceBlob)
        vec = embedder.forward()

        # perform classification to recognize the face
        preds = recognizer.predict_proba(vec)[0]
        j = np.argmax(preds)
        proba = preds[j]
        name = le.classes_[j]

        text = "{}: {:.2f}%".format(name, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)