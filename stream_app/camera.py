import os
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

from deepface import DeepFace


class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_classifier = cv2.CascadeClassifier(
            cv2.samples.findFile(cv2.data.haarcascades +
                                 'haarcascade_frontalface_default.xml')
        )
        self.classifier = load_model(os.path.join(
            os.getcwd(), 'model_mobilenet_small.h5'))
        # self.classifier = load_model('../stream_app/model_mobilenet_small.h5')
        self.class_labels = {
            0: 'angry',
            1: 'fear',
            2: 'happy',
            3: 'neutral'
        }

    def __del__(self):
        self.cap.release()

    # def get_frame(self):
    #     ret, frame = self.cap.read()
    #     frame_flip = cv2.flip(frame, 1)
    #     gray = cv2.cvtColor(frame_flip, cv2.COLOR_RGB2GRAY)
    #     faces = self.face_classifier.detectMultiScale(gray)
    #     for (x, y, w, h) in faces:
    #         roi_gray = gray[y: y + h, x: x + h]
    #         roi_gray = cv2.resize(
    #             roi_gray,
    #             (48, 48),
    #             interpolation=cv2.INTER_AREA
    #         )
    #         cv2.rectangle(
    #             frame_flip,
    #             (x, y),
    #             (x + h, y + h),
    #             (255, 0, 0),
    #             2
    #         )
    #         if np.sum([roi_gray]) != 0:
    #             roi = roi_gray.astype('float')/255.0
    #             roi = img_to_array(roi)
    #             roi = np.expand_dims(roi, axis=0)
    #             roi = np.repeat(roi, repeats=3, axis=-1)

    #             predictions = self.classifier.predict(roi)[0]
    #             # print("\nprediction = ", predictions)
    #             label = self.class_labels[predictions.argmax()]
    #             # print("\nprediction max = ", predictions.argmax())
    #             # print("\nlabel = ", label)
    #             label_position = (x, y)
    #             cv2.putText(
    #                 frame_flip,
    #                 label,
    #                 label_position,
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 2,
    #                 (0, 255, 0),
    #                 3
    #             )
    #             # print("\n\n")
    #     ret, frame = cv2.imencode('.jpg', frame_flip)
    #     return frame.tobytes()

    def get_frame(self):
        ret, frame = self.cap.read()
        frame_flip = cv2.flip(frame, 1)
        faces = self.face_classifier.detectMultiScale(frame_flip)
        for (x, y, w, h) in faces:
            roi = frame_flip[y: y + h, x: x + h]
            roi = cv2.resize(
                roi,
                (48, 48),
                interpolation=cv2.INTER_AREA
            )
            cv2.rectangle(
                frame_flip,
                (x, y),
                (x + h, y + h),
                (255, 0, 0),
                2
            )
        try:
            label = DeepFace.analyze(frame, actions=["emotion"])
            label_position = (x, y)
            cv2.putText(
                frame_flip,
                label["dominant_emotion"],
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                3
            )
        except:
            pass
        ret, frame = cv2.imencode('.jpg', frame_flip)
        return frame.tobytes()
