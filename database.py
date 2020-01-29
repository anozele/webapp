import matplotlib.pyplot as plt
import numpy as np
import pickle
import FaceToolKit as ftk
import DetectionToolKit as dtk
import os
verification_threshhold = 1.175
image_size = 160
v = ftk.Verification()
# Pre-load model for Verification
v.load_model("./models/20180204-160909/")
v.initial_input_output_tensors()
d = dtk.Detection()



def img_to_encoding(img):
    image = plt.imread(img)
    aligned = d.align(image, False)[0]
    return v.img_to_encoding(aligned, image_size)

face_database = {}

####here give dataset folder ....one time run code

for images in os.listdir('dataset'):
    identity = os.path.splitext(os.path.basename(images))[0]

    face_database[identity]=img_to_encoding(os.path.join('dataset',images))
    
file = open("face_database.pkl",'wb')
pickle.dump(face_database,file)
