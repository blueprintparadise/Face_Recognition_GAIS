# Importing essential files from the program
import Face_Recog
from Face_Recog import Main_Model
from Face_Recog.commons import functions
from Face_Recog import realtime
# Flask is used for webstreaming
from flask import Response
from flask import Flask
from flask import render_template
import threading
# Fancy Progress Bars
import cv2
from tqdm import tqdm

global offset
offset = False
import os
import pandas as pd
import threading
import requests

outputFrame = None
import time

lock = threading.Lock()
# from flask_swagger_ui import get_swaggerui_blueprint
# For certificate - converting "http" request to "https"
from OpenSSL import SSL

'''PLEASE NOTE ---------------------------------------------
    The min_detection_confidence in Mediapipe
    And 
    The  if w > 0 (line 73- realtime) - Need to be modified according to the hardware/Webcam used.
'''
outputFrame = None
time_buffer = True
# Needed to ensure streaming works on multiple devices
lock = threading.Lock()
app = Flask(__name__)
# app.register_blueprint(request_api.get_blueprint())
# Initializing Necessary model for recognition/detection
# Using "Facenet" and "Mediapipe" recommended
model_name = 'Facenet'

db_path = r"images"
detector_backend = 'opencv'
''' Options-'opencv',
         'ssd' ,
         'dlib',
         'mtcnn',
         'retinaface',
         'mediapipe'
'''
# distance_metric - used to judge distance between video_feed and database image
distance_metric = 'cosine'
input_shape = (224, 224)


@app.route("/Facevideo", methods=['GET', 'POST'])
def Facevideo():
    # cam = cv2.VideoCapture(0)
    #
    # try:
    #
    #     # creating a folder named data
    #     if not os.path.exists('data'):
    #         os.makedirs('data')
    #
    # # if not created then raise error
    # except OSError:
    #     print('Error: Creating directory of data')
    #
    # # frame
    # currentframe = 0
    #
    # while (True):
    #
    #     # reading from frame
    #     ret, frame = cam.read()
    #
    #     if ret:
    #         # if video is still left continue creating images
    #         name = './data/frame' + str(currentframe) + '.jpg'
    #         print('Creating...' + name)
    #
    #         # writing the extracted images
    #         cv2.imwrite(name, frame)
    #
    #         # increasing counter so that it will
    #         # show how many frames are created
    #         currentframe += 1
    #     else:
    #         break
    response = requests.Response()
    print(response)
    return "Runing"


# Embedding Images to dataframe
'''
[1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Rushi',
1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Rushi',
1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Rushi', 1, 'Akshay', 1, 'Akshay', 1, 'Akshay', 1, 'Akshay', 1, 'Akshay', 1,
'Rushi', 1, 'Rushi', 1, 'Akshay', 1, 'Rushi', 1, 'Rushi', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi',
'Akshay', 2, 'Rushi', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 2, 'Rushi', 'Akshay', 3,
'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay',
'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi',
'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3,
'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi',
'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay',
2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2,
'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi',
'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3,
'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay',
'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 2, 'Rushi',
'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Pranay', 'Akshay',
2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Pranay', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2,
'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi', 'Akshay', 2, 'Rushi',
'Akshay', 2, 'Rushi', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay',
'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Pranay', 'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Rushi',
'Akshay', 'Akshay', 3, 'Rushi', 'Akshay', 'Akshay', 3, 'Pranay', 'Akshay', 'Akshay', 3, 'Pawan', 'Akshay', 'Akshay', 2,
'Akshay', 'Akshay', 2, 'Akshay', 'Akshay', 2, 'Akshay', 'Akshay', 2, 'Akshay', 'Akshay', 3, 'Pawan', 'Akshay', 'Akshay',
3, 'Akshay', 'Akshay', 'Akshay', 3, 'Pawan', 'Akshay', 'Akshay', 3]
'''


def notification_logic():
    pass


@app.route('/name', methods=["GET"])
def get_name():
    lst = []
    string = ''
    name = realtime.api_notification()
    People_Count = [i for i in name if type(i) == int]
    num_of_people = People_Count[-2]
    num_of_people = num_of_people * 2
    Names = [i for i in name if type(i) == str]
    Names = Names[-num_of_people]
    print(Names)

    # final_x = int(final_x*2)
    # final_x_2= final_x_2[-final_x]

    return '01'


@app.route('/embed', methods=["GET"])
def embed(model_name, db_path, detector_backend, distance_metric):
    employees = []
    # check passed db folder exists
    if os.path.isdir(db_path):
        for r, d, f in os.walk(db_path):  # r=root, d=directories, f = files
            for file in f:
                if '.jpg' in file:
                    # exact_path = os.path.join(r, file)
                    exact_path = r + "/" + file
                    employees.append(exact_path)
    if len(employees) == 0:
        print("WARNING: There is no image in this path ( ", db_path, ") . Face recognition will not be performed.")
    model = Face_Recog.Main_Model.build_model(model_name)
    print(model_name, " is built")
    pbar = tqdm(range(0, len(employees)), desc='Finding embeddings')
    input_shape = Face_Recog.commons.functions.find_input_shape(model)
    input_shape_x = input_shape[0];
    input_shape_y = input_shape[1]

    embeddings = []
    # for employee in employees:
    for index in pbar:
        employee = employees[index]
        pbar.set_description("Finding embedding for %s" % (employee.split("/")[-1]))
        embedding = []

        # preprocess_face returns single face. this is expected for source images in db.
        img = functions.preprocess_face(img=employee, target_size=(input_shape_y, input_shape_x),
                                        enforce_detection=False, detector_backend=detector_backend)
        img_representation = model.predict(img)[0, :]

        embedding.append(employee)
        embedding.append(img_representation)
        embeddings.append(embedding)

    df = pd.DataFrame(embeddings, columns=['employee', 'embedding'])
    df['distance_metric'] = distance_metric
    # returns dataframe with employee, embedding and distance_metric information
    return df


@app.route('/retrn', methods=["GET"])
def retrn():
    emb = embed(model_name, db_path, detector_backend, distance_metric)
    lst = ''
    for i in emb['embedding']:
        lst = str(lst) + str(i)
    return lst


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(
        realtime.analysis(db_path, detector_backend=detector_backend, df=df, model_name=model_name, time_threshold=1,
                          frame_threshold=1, distance_metric="cosine"),
        mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    # start a thread that will perform web stream
    df = embed(model_name, db_path, detector_backend, distance_metric)
    t = threading.Thread()
    t.daemon = True
    print("System Running Succesfully")
    t.start()
    # start the flask app
    app.jinja_env.cache = {}
    app.run(host='0.0.0.0', port='8880', threaded=True, debug=False
            )
