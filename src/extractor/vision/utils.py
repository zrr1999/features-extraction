from __future__ import annotations

import pickle
from itertools import product
from typing import Any, Sequence

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def float2int(value: float, max_value: int) -> int:
    return relu(int(max_value * value))


def relu(value: int) -> int:
    return max(value, 0)


def get_face_by_dlib(image):
    import dlib

    detector = dlib.get_frontal_face_detector()  # type: ignore  # noqa: PGH003
    faces = detector(image)
    if faces:
        best_face = faces[0]
        return (
            relu(best_face.left()),
            relu(best_face.top()),
            relu(best_face.width()),
            relu(best_face.height()),
        )


def get_face_by_mediapipe(image):
    from mediapipe.python.solutions import face_detection

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results: Any = face_detection.process(rgb_image)
    if results.detections:
        best_detection = results.detections[0]
        # drawing_utils.draw_detection(image, best_detection)
        box = best_detection.location_data.relative_bounding_box
        return (
            float2int(box.xmin, image.shape[1]),
            float2int(box.ymin, image.shape[0]),
            float2int(box.width, image.shape[1]),
            float2int(box.height, image.shape[0]),
        )


def get_face_by_ssd(image):
    # 加载预训练的 SSD 模型
    import numpy as np

    model_file = "./ssd/res10_300x300_ssd_iter_140000.caffemodel"
    config_file = "./ssd/deploy.prototxt"
    model = cv2.dnn.readNetFromCaffe(config_file, model_file)
    height, width = image.shape[:2]
    # 构建输入图像的 blob
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0))
    # 设置模型输入
    model.setInput(blob)
    # 进行推理
    detections = model.forward()
    # 提取人脸检测结果
    face_coords = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # 设置置信度阈值
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            x, y, x1, y1 = box.astype(int)  # 左上角和右下角坐标
            # print(f"Detected face coordinates: x={x}, y={y}, width={w-x}, height={h-y}")
            return (x, y, x1 - x, y1 - y)


def get_face_by_mtcnn(image):
    from mtcnn import MTCNN

    detector = MTCNN()
    faces = detector.detect_faces(image)
    # print ("-------------------")
    if faces:
        best_face = faces[0]["box"]
        x, y, w, h = best_face
        # print(f"Detected face coordinates: x={x}, y={y}, width={w}, height={h}")
        return x, y, w, h


def detect_face(image, detector: str = "dlib"):
    if detector == "dlib":
        return get_face_by_dlib(image)
    elif detector == "mediapipe":
        return get_face_by_mediapipe(image)
    elif detector == "mtcnn":
        return get_face_by_mtcnn(image)
    elif detector == "ssd":
        return get_face_by_ssd(image)
    else:
        raise ValueError(f"Unknown detector: {detector}")


def extract_face_features(image, model_name: str = "Facenet"):
    from deepface import DeepFace

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_features = DeepFace.represent(image, model_name=model_name, detector_backend="skip")
    return face_features[0]["embedding"]


def get_all_vision_methods(ignore_methods: Sequence[str] = ()):
    detection_methods = ["dlib", "mediapipe", "ssd", "mtcnn"]
    recognition_methods = [
        "Facenet",
        "ArcFace",
        "OpenFace",
        "DeepFace",
        "Facenet512",
        "VGG-Face",
        # "DeepID",
    ]
    feature_size = {
        "Facenet": 128,
        "ArcFace": 512,
        "OpenFace": 128,
        "DeepFace": 4096,
        "Facenet512": 512,
        "VGG-Face": 2622,
    }

    for detection_method, recognition_method in product(detection_methods, recognition_methods):
        if detection_method in ignore_methods or recognition_method in ignore_methods:
            continue
        yield detection_method, recognition_method, feature_size[recognition_method]
