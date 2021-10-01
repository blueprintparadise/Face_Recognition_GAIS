from deepface import DeepFace
from deepface.commons import functions
from flask import Response
from flask import Flask
from flask import render_template
import threading
from tqdm import tqdm
import os
import pandas as pd
from OpenSSL import SSL
# context = SSL.Context(SSL.PROTOCOL_TLSv1_2)
# context.use_privatekey_file('server.key')
# context.use_certificate_file('server.crt')
outputFrame = None
import realtime

lock = threading.Lock()
app = Flask(__name__)
model_name = 'Facenet'
db_path = r"./images"
detector_backend = 'mediapipe'
distance_metric = 'euclidean'
input_shape = (224, 224)
#print(distance_metric)

def embed(model_name, db_path, detector_backend, input_shape, distance_metric):

    employees = []
    # check passed db folder exists
    if os.path.isdir(db_path) == True:
        for r, d, f in os.walk(db_path):  # r=root, d=directories, f = files
            for file in f:
                if ('.jpg' in file):
                    # exact_path = os.path.join(r, file)
                    exact_path = r + "/" + file
                    # print(exact_path)
                    employees.append(exact_path)
    if len(employees) == 0:
        print("WARNING: There is no image in this path ( ", db_path, ") . Face recognition will not be performed.")
    if len(employees) > 0:
        model = DeepFace.build_model(model_name)
        print(model_name, " is built")
    pbar = tqdm(range(0, len(employees)), desc='Finding embeddings')
    input_shape = functions.find_input_shape(model)
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
    return df


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(realtime.analysis(db_path,
                                      enable_face_analysis=False, detector_backend=detector_backend, df=df,model_name=model_name),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    # start a thread that will perform web stream
    df = embed(model_name, db_path, detector_backend, input_shape, distance_metric)
    t = threading.Thread()
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host='0.0.0.0', port='8001', debug=True,
            use_reloader=False,ssl_context='adhoc')

# (r"C:\Users\user\Documents\GAIS\Face-recognition-Using-Facenet-On-Tensorflow-2.X\Faces",
#                enable_face_analysis=False,detector_backend='mediapipe')
