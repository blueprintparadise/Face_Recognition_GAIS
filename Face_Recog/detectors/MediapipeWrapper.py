from Face_Recog.detectors import FaceDetector


# Link - https://google.github.io/mediapipe/solutions/face_detection

def build_model():
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
    # min_detection_confidence - "A filter to analyse the training photographs"
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.50)
    # Returns detected face
    return face_detection


def detect_face(face_detector, img, align=True):
    import mediapipe as mp
    import re
    # Regular expressions
    # mp_face_detection = mp.solutions.face_detection
    resp = []
    results = face_detector.process(img)
    original_size = img.shape
    target_size = (300, 300)
    # First face , than eye
    if results.detections:
        for detection in results.detections:
            # mp_drawing.draw_detection(img, detection)
            # detected_face is the cropped image that is then passed forward to the Regognizer
            '''
            DETECTION - 
            Collection of detected faces, where each face is represented as a detection proto message that contains 
            a bounding box and 6 key points (right eye, left eye, nose tip, mouth center, right ear tragion, and left
            ear tragion). The bounding box is composed of xmin and width (both normalized to [0.0, 1.0] by the
            image width) and ymin and height (both normalized to [0.0, 1.0] by the image height). Each key point
            is composed of x and y, which are normalized to [0.0, 1.0] by the image width and height
            respectively.
            '''
            # Bounding Box
            x = re.findall('xmin: (..*)', str(detection))
            y = re.findall('ymin: (..*)', str(detection))
            h = re.findall('height: (..*)', str(detection))
            w = re.findall('width: (..*)', str(detection))
            # Eye Locations
            reye_x = re.findall('x: (..*)', str(detection))[0]
            leye_x = re.findall('x: (..*)', str(detection))[1]
            reye_y = re.findall('y: (..*)', str(detection))[0]
            leye_y = re.findall('y: (..*)', str(detection))[1]
            # Detections are normalized by the mediapipe API, thus they need to be multiplied
            # Extra tweaking done to improve accuracy
            x = (float(x[0]) * original_size[1])-10
            y = (float(y[0]) * original_size[0])-40
            h = (float(h[0]) * original_size[0])+30
            w = (float(w[0]) * original_size[1])+20
            reye_x = (float(reye_x) * original_size[1])
            leye_x = (float(leye_x) * original_size[1])
            reye_y = (float(reye_y) * original_size[0])
            leye_y = (float(leye_y) * original_size[0])
            if float(x) and float(y) > 0:
                detected_face = img[int(y):int(y + h), int(x):int(x + w)]
                img_region = [int(x), int(y), int(w), int(h)]
                if align:
                    left_eye = (leye_x, leye_y)
                    right_eye = (reye_x, reye_y)
                    detected_face = FaceDetector.alignment_procedure(detected_face, left_eye, right_eye)
                resp.append((detected_face, img_region))
            else:
                continue

    # resp is a tuple containing the detected face and the area in the image the face exists
    return resp

