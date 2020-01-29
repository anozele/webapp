import matplotlib.pyplot as plt
import numpy as np
import os
import FaceToolKit as ftk
import DetectionToolKit as dtk
import pickle
from flask import Flask, request
from flask_bootstrap import Bootstrap
from flask_restful import Resource, Api
from flask import Flask, render_template, redirect, url_for
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from werkzeug import secure_filename
import cv2
import io
import base64

app = Flask(__name__)
import flask
bootstrap = Bootstrap(app)


verification_threshhold = 1.175
image_size = 160
v = ftk.Verification()
# Pre-load model for Verification
v.load_model("./models/20180204-160909/")
v.initial_input_output_tensors()

d = dtk.Detection()



def build_graph(image):
    img = io.BytesIO()
    plt.imshow(image)
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return 'data:image/png;base64,{}'.format(graph_url)


def distance(emb1, emb2):
    diff = np.subtract(emb1, emb2)
    return np.sum(np.square(diff))

def who_is_it(encoding, database):
    
    min_dist = 1000
    for (name, db_enc) in database.items():
        print("db_enc",db_enc)
        dist = distance(encoding, db_enc)
        if min_dist > dist:
            min_dist = dist
            identity = name
            
    
    fig = plt.figure(figsize=(10,10))
    if min_dist > verification_threshhold:
        print("Not in the database.")
        identity = "Not in the database."
    else:
        identity = str(identity)
        print ("it's " + str(identity) + ", Dissimlarity =" + str(min_dist) )
        
    return  identity

def add_to_database(images,person_name,database):
    db={}
    identity = person_name
    encodings_list = img_to_encoding(images)
    #db[identity]=img_to_encoding(images)
    for encodings in encodings_list:
        for encod in encodings[0]:
            db[identity]=encod

    database.update(db)

    f=open('face_database.pkl', 'wb')
    pickle.dump(database, f)
    f.close()
    return database
    
def img_to_encoding(img_list):
    encodings_list=[]
    for img in img_list:
        encodings=[]
        try:
            image = plt.imread(img)
            print("image",image)
        except Exception as e:
            print(e)
        faces = d.align(image, True)
        print("aligned",faces)
        for face in faces:
            print(face)
            encodings.append(np.array(v.img_to_encoding(face,image_size),dtype="float32").reshape(1,-1))
        encodings_list.append([encodings,image])

    print("encodings_list",encodings_list)
    print(len(encodings_list))
    return encodings_list



def test_model(path_to_image,database):
    prediction_list=[]
    encodings_list = img_to_encoding(path_to_image)
    count=0
    for encodings in encodings_list:
        pred=[]
        print("encodings",encodings)
        #test=np.array(encodings,dtype="float32").reshape(1, -1)
        #plt.imshow(path_to_image)
        #print(test.shape)
        for encod in encodings[0]:
            #print("encod",len(encod))
            pred.append([who_is_it(encod,database)])
            print("pred",pred)
        prediction_list.append([build_graph(encodings[1]),pred[0]])
        count+=1
    #print("prediction_list",prediction_list)
    return prediction_list



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        #file = request.files['file']
        uploaded_files =  flask.request.files.getlist("file")
        person_name  = request.form.get('uname')
        #sfname = 'static/images/'+str(secure_filename(uploaded_files.filename))
        #print("uploadfile",uploadfile)
        if uploaded_files:
            try:
                filedb= open("face_database.pkl",'rb')
                database = pickle.load(filedb)
                add_to_database(uploaded_files,person_name,database)
                return "sucessfully registered"
            except Exception as e:
                print(e)
                return "Oops something gone wrong !!!!!"
        else:
            return "please ulpoad an image "
    return render_template("register.html")
    

@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        #file = request.files['file']
        uploaded_files =  flask.request.files.getlist("file")
        #sfname = 'static/images/'+str(secure_filename(uploaded_files.filename))
        #print("uploadfile",uploadfile)
        if uploaded_files:
            filedb= open("face_database.pkl",'rb')
            database = pickle.load(filedb)
            return render_template("pic.html",results=[graph for graph in test_model(uploaded_files,database)])
        else:
            return "please ulpoad an image "
    
    return render_template("login.html")


if __name__ == '__main__':
   app.run(debug = True)

