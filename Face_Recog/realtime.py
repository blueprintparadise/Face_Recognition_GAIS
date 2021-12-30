import os
import datetime
import time
#image_path = os.environ['images']
#device_token = os.environ['token']
import mysql.connector
from tqdm import tqdm
import numpy as np
import math
with open('Face_Recog/myfile.txt', 'w') as fp:
    x = datetime.datetime.now()
    print(str(x.strftime("%X")))
    fp.write(str(x.strftime("%X")))
file1 = open(r"Face_Recog/myfile.txt","r+")
import cv2
import time
import re
import os
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Face_Recog import Main_Model
from Face_Recog.commons import functions, distance as dst
from Face_Recog.detectors import FaceDetector
from scipy.spatial import distance as dist
from tensorflow.keras.models import model_from_json
import threading
#import pafy
import requests
notification_time = 5
Threshold_setter = 0.5
from Face_Recog import  Liveness_Blinking
Blink_time = 30
name_list = []
from urllib3.exceptions import InsecureRequestWarning
mydb = mysql.connector.connect(host="43.231.127.150",port = "3306", user="usha-jagadambe", passwd="Gais@2020$usha-jagadambe", database="securitydb",
                                           auth_plugin='mysql_native_password')

# Suppress only the single warning from urllib3 needed.
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

import asyncio
#my_token  = "f0QJitMCQc6UHtt2pJWa7S:APA91bESm9UDipNbPPA4SG-unn-e6VFInm15Rp4O5IaiaBrcjbBTzEoFzymx2CfkiPwaxaIo5d_wgvWZbOpYFF2Mzmmm_eZWMtYwhQ2JbQc9o7MzbyMOk6H7kUQi4Q7lPE0lBnRDMRxS"
url = "https://43.231.127.150:7788/api/notification/sendnotificationtodevice?\
deviceToken={}/\
&message={}&\
title={}"

def get_token():
    mycursor = mydb.cursor(buffered=True)
    query = "select token from tb_user where id=1"
    mycursor.execute(query)
    row = mycursor.fetchone()
    mycursor.close()
    mydb.commit()

    return  row

def convert(date_time):
    format = '%X'  # The format
    datetime_str = datetime.datetime.strptime(date_time, format)
    return datetime_str


def Listing(name):
    name_list.append(name)
    return None
def api_notification():
    name = name_list[-1]
    return str(name)
eof = '''2, 'images\\\\Rushi', 'images\\\\Akshay', 2, 'images\\\\Rushi', 'images\\\\Akshay', 2, 'images\\\\Rushi', 'images\\\\Rajan', 1]
['images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 
'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Rajan', 'images\\\\Rushi', 'images\\\\Rajan', 'images\\\\Rushi', 'images\\\\Rajan', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Rajan', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Rajan', 'images\\\\Rushi', 'images\\\\Rajan', 'images\\\\Rushi', 'images\\\\Rajan', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Rajan', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Rajan', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Akshay', 'images\\\\Rushi', 'images\\\\Rajan']'''

# read file and text

def get_name():
    if len(name_list)>20:
            print("Inside Get_Name")
            People_Count = [i for i in name_list if type(i) == int]
            num_of_people = People_Count[-2]
            num_of_people = num_of_people * 2
            Names = [i for i in name_list if type(i) == str]
            new_names = []
            for i in Names:
                new = i.replace("images\\\\", "")
                new_names.append(new)
            fin_names = new_names[-num_of_people:]
            uniqueNames = set(fin_names)
            upd_names = list(uniqueNames)
            #-------------------------------------------------------------------------------------------------------------
            the_token = get_token()
            print(the_token)
            url_post = url.format(str(the_token),"Has Arrived",upd_names)
            file1 = open("Face_Recog/myfile.txt", "r+")
            x = file1.read()
            now = datetime.datetime.now().strftime("%X")
            c = convert(str(now)) - convert(str(x))
            if float(c.total_seconds())>10:

                r = requests.post(url = url_post,verify=False)
                if r.status_code == 200:
                    file1.seek(0)
                    file1.truncate(0)
                    file1.write(str(datetime.datetime.now().strftime("%X")))
                    print(file1.read())

        # Logic for pushing the notification
        #print(Names)

    return None
def analysis(db_path, df, model_name='Facenet', detector_backend='mediapipe', distance_metric='cosine',
             source=0, time_threshold=5, frame_threshold=5,enable_multiple=True):
    blinklist = []
    face_detector = FaceDetector.build_model(detector_backend)
    print("Detector backend is ", detector_backend)
    tic = time.time()
    # ------------------------

    employees = []
    # check passed db folder exists
    if os.path.isdir(db_path) == True:
        for r, d, f in os.walk(db_path):  # r=root, d=directories, f = files
            for file in f:
                if ('.jpg' in file):
                    # exact_path = os.path.join(r, file)
                    exact_path = r + "/" + file
                    employees.append(exact_path)

    if len(employees) == 0:
        print("WARNING: There is no image in this path ( ", db_path, ") . Face recognition will not be performed.")

    if len(employees) > 0:
        model = Main_Model.build_model(model_name)
        print(model_name, " is built")

        input_shape = functions.find_input_shape(model)
        input_shape_x = input_shape[0];
        input_shape_y = input_shape[1]

        # tuned thresholds for model and metric pair
        threshold = dst.findThreshold(model_name, distance_metric)

    pivot_img_size = 112  # face recognition result image
    freeze = False
    face_detected = False
    face_included_frames = 0  # freeze screen if face detected sequantially 5 frames
    freezed_frame = 0
  #-----------------------------------------------------------------------------------------------------------------------------------------------
    cap = cv2.VideoCapture(0)  # webcam
    while (True):
        ret, img = cap.read()
        Blink = Liveness_Blinking.Liveness(ret = ret, frame = img)
        #print(Blink)
        if img is None:
            break
        raw_img = img.copy()
        if freeze == False:
            # faces stores list of detected_face and region pair
            faces = FaceDetector.detect_faces(face_detector, detector_backend, img, align=True)
            print(len(faces))
            Listing(len(faces))

            if len(faces) == 0:
                face_included_frames = 0
        else:
            num_faces = 1 if not enable_multiple else len(faces)
            #num_actions = len(actions)
            #num_checks = num_actions * num_faces
            #disable_option = num_checks <= 1 or not prog_bar
            #pbar = tqdm(range(0, num_checks), desc='Processing Faces', disable=disable_option)
            #print(num_faces)
            face_response_obj = {}

            faces = []
        detected_faces = []
        face_index = 0
        for face, (x, y, w, h) in faces:
            if w > 0:  # discard small detected faces
                face_detected = True
                if face_index == 0:
                    face_included_frames = face_included_frames + 1  # increase frame for a single face
                detected_faces.append((x, y, w, h))
                face_index = face_index + 1

        if face_detected == True and face_included_frames == frame_threshold and freeze == False:
            freeze = True
            base_img = raw_img.copy()
            detected_faces_final = detected_faces.copy()


            if freezed_frame == 0:

                freeze_img = base_img.copy()
                Fin_img = base_img.copy()
                # freeze_img = np.zeros(resolution, np.uint8) #here, np.uint8 handles showing white area issue
                for detected_face in detected_faces_final:
                    x = detected_face[0];
                    y = detected_face[1]
                    w = detected_face[2];
                    h = detected_face[3]
                    cv2.rectangle(Fin_img, (x, y), (x + w, y + h), (67, 67, 67),
                                  1)  # draw rectangle to main image

                    # apply deep learning for custom_face
                    #live_img = base_img.copy()
                    #
                    custom_face = base_img[y:y + h, x:x + w]

                    #liveornot = Liveness_Detection(live_img, x, y, h, w)
                    # face recognition

                    custom_face = functions.preprocess_face(img=custom_face,
                                                            target_size=(input_shape_y, input_shape_x),
                                                            enforce_detection=False, detector_backend='mediapipe')

                    # check preprocess_face function handled
                    if custom_face.shape[1:3] == input_shape:
                        if df.shape[0] > 0:  # if there are images to verify, apply face recognition
                            img1_representation = model.predict(custom_face)[0, :]

                            def findDistance(row):
                                distance_metric = row['distance_metric']
                                img2_representation = row['embedding']

                                distance = 1000  # initialize very large value
                                if distance_metric == 'cosine':
                                    distance = dst.findCosineDistance(img1_representation, img2_representation)
                                elif distance_metric == 'euclidean':
                                    distance = dst.findEuclideanDistance(img1_representation, img2_representation)
                                elif distance_metric == 'euclidean_l2':
                                    distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation),
                                                                         dst.l2_normalize(img2_representation))

                                return distance

                            df['distance'] = df.apply(findDistance, axis=1)
                            df = df.sort_values(by=["distance"])

                            candidate = df.iloc[0]
                            employee_name = candidate['employee']
                            best_distance = candidate['distance']
                            values_ = candidate[['employee', 'distance']].values
                            name = employee_name
                            #print(name)
                            Listing(name)
                            #print("--------------------------------------------------------------")
                            if best_distance <= threshold - Threshold_setter:

                                display_img = cv2.imread(employee_name)

                                display_img = cv2.resize(display_img, (pivot_img_size, pivot_img_size))

                                label = employee_name.split("/")[-1].replace(".jpg", "")
                                label = re.sub('[0-9]', '', label)

                                cv2.rectangle(Fin_img, (10, 10), (210, 50), (67, 67, 67), -10)
                                cv2.putText(Fin_img, str("Unknown"), (20, 40),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
            toc = time.time()
            #print(blinklist)
            timer = toc - tic
            if int(timer) % 30 == 0 and sum(blinklist)<4:
                #print("_____________FAKE______________")
                blinklist=[]
            if Blink > 0:
                Blink = "BLINKING"
            else:
                Blink = "NOT BLINKING"
            cv2.rectangle(Fin_img, (10, 10), (400, 50), (67, 67, 67), -10)
            #cv2.putText(Fin_img, str(str(name)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            #            1, (255, 255, 255), 1)

            threading.Thread(target=get_name).start()
            if Blink == "BLINKING":
                Blink=1
            else:
                Blink=0
            ret, buffer = cv2.imencode('.jpg', Fin_img)
            frame = buffer.tobytes()
            freezed_frame = freezed_frame + 1

            #print(int(timer))
            blinklist.append(Blink)

# split analysis into two parts and return necessary stuff from



            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            freezed_frame = freezed_frame + 1
        else:
            face_detected = False
            face_included_frames = 0
            freeze = False
            freezed_frame = 0
            ret, buffer = cv2.imencode('.jpg', raw_img)
            frame2 = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame2 + b'\r\n')

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
            pass
    # kill open cv things
    cap.release()
    cv2.destroyAllWindows()
