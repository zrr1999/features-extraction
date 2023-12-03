import os
import cv2
from tools.utils import detect_face, extract_face_features
import pickle


dataset_path = "/home/zrr/workspace/face-recognition/datasets/Face-Dataset/UCEC-Face"
features_path = "/home/zrr/workspace/face-recognition/datasets/features"
detection_methods = ["dlib", "mediapipe"]
recognition_methods = ["Facenet", "ArcFace", "OpenFace", "DeepFace"]

os.makedirs(features_path, exist_ok=True)

for detection_method, recognition_method in zip(detection_methods, recognition_methods):
    data = {}
    for i, j in zip(range(1, 131), range(60)):
        subject_path = f"{dataset_path}/subject{i}"
        image_path = f"{subject_path}/subject{i}.{j}.png"
        input_image = cv2.imread(image_path)
        output_image = input_image.copy()

        face = detect_face(input_image, detector=detection_method)

        if face:
            x, y, w, h = face
            face_image = input_image[y:y + h, x:x + w]
            features = extract_face_features(face_image, model_name=recognition_method)
            data[(i, j)] = (image_path, features)

    with open(f"{features_path}/{detection_method}_{recognition_method}.pkl", "wb") as file:
        pickle.dump(data, file)
    
    with open(f"{features_path}/{detection_method}_{recognition_method}.pkl", 'rb') as file:
        loaded_data = pickle.load(file)
        assert data == loaded_data
