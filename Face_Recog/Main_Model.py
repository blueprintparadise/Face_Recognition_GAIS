import warnings

warnings.filterwarnings("ignore")

import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from Face_Recog.basemodels import VGGFace, OpenFace, Facenet, Facenet512, FbDeepFace, DeepID, DlibWrapper, ArcFace, \
    Boosting
from Face_Recog.extendedmodels import Age, Gender, Race, Emotion
from Face_Recog.commons import functions, distance as dst

import tensorflow as tf

tf_version = int(tf.__version__.split(".")[0])
if tf_version == 2:
    import logging

    tf.get_logger().setLevel(logging.ERROR)


def build_model(model_name):
    """
	This function builds a deepface model
	Parameters:
		model_name (string): face recognition or facial attribute model
			VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
			Age, Gender, Emotion, Race for facial attributes

	Returns:
		built deepface model
	"""

    global model_obj  # singleton design pattern

    models = {
        'VGG-Face': VGGFace.loadModel,
        'OpenFace': OpenFace.loadModel,
        'Facenet': Facenet.loadModel,
        'Facenet512': Facenet512.loadModel,
        'DeepFace': FbDeepFace.loadModel,
        'DeepID': DeepID.loadModel,
        'Dlib': DlibWrapper.loadModel,
        'ArcFace': ArcFace.loadModel,
        'Emotion': Emotion.loadModel,
        'Age': Age.loadModel,
        'Gender': Gender.loadModel,
        'Race': Race.loadModel
    }

    if not "model_obj" in globals():
        model_obj = {}

    if not model_name in model_obj.keys():
        model = models.get(model_name)
        if model:
            model = model()
            model_obj[model_name] = model
        # print(model_name," built")
        else:
            raise ValueError('Invalid model_name passed - {}'.format(model_name))

    return model_obj[model_name]


functions.initializeFolder()
