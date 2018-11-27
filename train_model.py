# -*- coding: utf-8 -*-

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# ---------------------------
# --embeddings : The path to the serialized embeddings (we exported it by running the previous extract_embeddings.py  script).
# --recognizer : This will be our output model that recognizes faces. It is based on SVM. We’ll be saving it so we can use it in the next two recognition scripts.
# --le : Our label encoder output file path. We’ll serialize our label encoder to disk so that we can use it and the recognizer model in our image/video face recognition scripts.
import json
ap = argparse.ArgumentParser()
ap.add_argument("-cfg", "--config", default = "train_model.json", help = "Path to the config file (default one is extract_embeddings.json)")
args = vars(ap.parse_args())

cfg = json.loads(open(args["config"]).read())
# ---------------------------

print("[INFO] Loading face embeddings ...")
data = pickle.loads(open(cfg["embeddings"], "rb").read())

print("[INFO] encoding labels ...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open(cfg["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()
 
# write the label encoder to disk
f = open(cfg["le"], "wb")
f.write(pickle.dumps(le))
f.close()