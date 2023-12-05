import cv2
from typing import Any
import torch
import pickle
from itertools import product
from typing import Sequence
from torch.utils.data import Dataset, DataLoader
from torch import nn


def float2int(value: float, max_value: int) -> int:
    return relu(int(max_value * value))


def relu(value: int) -> int:
    return max(value, 0)


def get_face_by_dlib(image):
    import dlib

    detector = dlib.get_frontal_face_detector()  # type: ignore
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


def detect_face(image, detector: str = "dlib"):
    if detector == "dlib":
        return get_face_by_dlib(image)
    elif detector == "mediapipe":
        return get_face_by_mediapipe(image)
    else:
        raise ValueError(f"Unknown detector: {detector}")


def extract_face_features(image, model_name: str = "Facenet"):
    from deepface import DeepFace

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_features = DeepFace.represent(
        image, model_name=model_name, detector_backend="skip"
    )
    return face_features[0]["embedding"]


def get_example_image():
    dataset_path = "/home/zrr/workspace/face-recognition/datasets"
    input_image = cv2.imread(
        f"{dataset_path}/Face-Dataset/UCEC-Face/subject1/subject1.4.png"
    )
    return input_image


def load_features(features_path: str, detection_method:str, recognition_method:str):
    file_path = f"{features_path}/{detection_method}_{recognition_method}.pkl"
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def calculate_accuracy(model: nn.Module, data_loader: DataLoader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in data_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)  # 获取最大概率的预测结果
            total += labels.size(0)  # 更新总样本数
            correct += (predicted == labels).sum().item()  # 更新正确预测的样本数

    accuracy = 100 * correct / total
    return accuracy


def calculate_class_weights(data_loader: DataLoader, num_classes: int=130):
    class_counts = [0] * num_classes

    for _, labels in data_loader:
        for i in range(num_classes):
            class_counts[i] += (labels == i).sum().item()
    
    total_samples = sum(class_counts)
    class_weights = []
    for i in range(num_classes):
        class_weights.append(class_counts[i] / total_samples)

    return class_weights

def calculate_f1_score(model: nn.Module, data_loader: DataLoader):
    class_weights = calculate_class_weights(data_loader)
    num_classes = len(class_weights)
    class_f1_scores = [0] * num_classes
    class_counts = [0] * num_classes

    for features, labels in data_loader:
        outputs = model(features)
        _, predicted = torch.max(outputs.data, 1)

        for i in range(num_classes):
            true_positives = ((predicted == i) & (labels == i)).sum().item()
            false_positives = ((predicted == i) & (labels != i)).sum().item()
            false_negatives = ((predicted != i) & (labels == i)).sum().item()

            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)

            f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

            class_f1_scores[i] += f1_score
            class_counts[i] += 1

    weighted_f1_score = 0

    for i in range(num_classes):
        class_weight = class_weights[i]
        class_f1 = class_f1_scores[i] / class_counts[i]
        weighted_f1_score += class_weight * class_f1
    return 100 * weighted_f1_score

def get_all_methods(ignore_methods: Sequence[str] = ()):
    detection_methods = ["dlib", "mediapipe"]
    recognition_methods = [
        "Facenet",
        "ArcFace",
        "OpenFace",
        "DeepFace",
        "Facenet512",
        "VGG-Face",
    ]
    feature_size = {
        "Facenet": 128,
        "ArcFace": 512,
        "OpenFace": 128,
        "DeepFace": 4096,
        "Facenet512": 512,
        "VGG-Face": 2622,
    }

    for detection_method, recognition_method in product(
        detection_methods, recognition_methods
    ):
        if detection_method in ignore_methods or recognition_method in ignore_methods:
            continue
        yield detection_method, recognition_method, feature_size[recognition_method]
