import os
import cv2
import dlib
import pickle
import config
import numpy as np
import tensorflow as tf
from imutils import face_utils
from siameseNetwork import get_siamese_model, preprocess_input

config.model = get_siamese_model()
config.face_detector = dlib.get_frontal_face_detector()


def predict_people(image):
    name = 'not known'
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = config.face_detector(gray, 0)
    for face in faces:
        face_bounding_box = face_utils.rect_to_bb(face)
        if all(i >= 0 for i in face_bounding_box):
            [x, y, w, h] = face_bounding_box
            frame = image[y:y + h, x:x + w]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            frame = cv2.resize(frame, (224, 224))
            frame = np.asarray(frame, dtype=np.float64)
            frame = np.expand_dims(frame, axis=0)
            frame = preprocess_input(frame)
            feature = config.model.get_features(frame)

            dist = tf.norm(config.features - feature, axis=1)
            name = 'not known'
            loc = tf.argmin(dist)
            if dist[loc] < 0.8:
                name = config.people[loc]
                    
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, name, (x, y-5), font_face, 0.8, (0,0,255), 3)
    return image

def generate_dataset_festures():
    people = []
    features = []
    dumpable_features = {}

    pickle_file = os.path.join(config.feature_dir, 'weights.pkl')
    if os.path.isfile(pickle_file):
        people, features, dumpable_features = load_pickle_file(pickle_file)
    
    image_dir_people = os.listdir(config.data_dir)
    for name in image_dir_people:
        if name not in people:
            nparr = generate_image_features(os.path.join(config.data_dir, name))
            features.append(nparr)
            people.append(name)
            dumpable_features[name] = nparr
    
    config.features = features
    config.people = people

    dump_pickle_file(pickle_file, dumpable_features)
    print("Model Dumpped !!")


def generate_image_features(directory):
    images = []
    for image in os.listdir(directory):
        image_path = os.path.join(directory, image)
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = config.face_detector(gray, 0)
        if len(faces) == 0:
            continue
        for face in [faces[0]]:
            face_bounding_box = face_utils.rect_to_bb(face)
            if all(i >= 0 for i in face_bounding_box):
                [x, y, w, h] = face_bounding_box
                frame = img[y:y + h, x:x + w]
                frame = cv2.resize(frame, (224, 224))
                frame = np.asarray(frame, dtype=np.float64)
                images.append(frame)
    images = np.asarray(images)
    images = preprocess_input(images)
    images = tf.convert_to_tensor(images)
    feature = config.model.get_features(images)
    feature = tf.reduce_mean(feature, axis=0)
    return feature


def load_pickle_file(pickle_file):
    with open(pickle_file, 'rb') as f:
        dumpable_features = pickle.load(f)

    people = []
    features = []
    for key, value in dumpable_features.items():
        people.append(key)
        features.append(value)
    return people, features, dumpable_features


def dump_pickle_file(pickle_file, dumpable_features):
    if len(list(dumpable_features.keys())) > 0:
        with open(pickle_file, 'wb') as f:
            pickle.dump(dumpable_features, f)
