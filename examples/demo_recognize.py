import cv2
from tools.utils import get_face_by_dlib, extract_face_features


dataset_path = "/home/zrr/workspace/face-recognition/datasets"
input_image = cv2.imread(
    f"{dataset_path}/Face-Dataset/UCEC-Face/subject1/subject1.4.png"
)
output_image = input_image.copy()


face = get_face_by_dlib(input_image)
if face:
    x, y, w, h = face
    face_image = output_image[y : y + h, x : x + w]
    features = extract_face_features(face_image, model_name="Facenet")
    print(len(features))
    features = extract_face_features(face_image, model_name="ArcFace")
    print(len(features))
    features = extract_face_features(face_image, model_name="OpenFace")
    print(len(features))
    features = extract_face_features(face_image, model_name="DeepFace")
    print(len(features))
    # features = extract_face_features(face_image, model_name="VGG-Face")
    # print(len(features))
