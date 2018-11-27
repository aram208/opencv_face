# -*- coding: utf-8 -*-
from imutils import paths
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

# ---------------------------
# --dataset : The path to our input dataset of face images.
# --embeddings : The path to our output embeddings file. Our script will compute face embeddings which we’ll serialize to disk.
# --detector : Path to OpenCV’s Caffe-based deep learning face detector used to actually localize the faces in the images.
# --embedding-model : Path to the OpenCV deep learning Torch embedding model. This model will allow us to extract a 128-D facial embedding vector.
# --confidence : Optional threshold for filtering week face detections.
import json
ap = argparse.ArgumentParser()
ap.add_argument("-cfg", "--config", default = "extract_embeddings.json", help = "Path to the config file (default one is extract_embeddings.json)")
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

print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(cfg["dataset"]))

knownEmbeddings = []
knownNames = []

total = 0

for (i, imagePah) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
    name = imagePah.split(os.path.sep)[-2]

    image = cv2.imread(imagePah)
    image = imutils.resize(image, width = 600)
    (h, w) = image.shape[:2]
    
    imageBlob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)

    detector.setInput(imageBlob)
    detections = detector.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > cfg["confidence"]:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB = True, crop = False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

print("[INFO] serializing {} encodings ...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(cfg["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()