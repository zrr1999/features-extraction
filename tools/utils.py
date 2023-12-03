import cv2
from typing import Any


def float2int(value: float, max_value: int) -> int:
    return int(max_value * value)


def relu(value: int) -> int:
    return max(value, 0)

def get_face_by_dlib(image):
    import dlib
    detector = dlib.get_frontal_face_detector() # type: ignore
    faces = detector(image)
    if faces:
        best_face = faces[0]
        return relu(best_face.left()), relu(best_face.top()), relu(best_face.width()), relu(best_face.height())

def get_face_by_mediapipe(image):
    from mediapipe.python.solutions import face_detection
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results: Any = face_detection.process(rgb_image)
    if results.detections:
        best_detection = results.detections[0]
        # drawing_utils.draw_detection(image, best_detection)
        box = best_detection.location_data.relative_bounding_box
        print(box)
        return float2int(box.xmin, image.shape[1]), float2int(box.ymin, image.shape[0]), float2int(box.width, image.shape[1]), float2int(box.height, image.shape[0])

def detect_face(image, detector: str="dlib"):
    if detector == "dlib":
        return get_face_by_dlib(image)
    elif detector == "mediapipe":
        return get_face_by_mediapipe(image)
    else:
        raise ValueError(f"Unknown detector: {detector}")

def extract_face_features(image, model_name: str="Facenet"):
    from deepface import DeepFace
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_features = DeepFace.represent(image, model_name=model_name, detector_backend="skip")
    return face_features[0]["embedding"]

def get_example_image():
    dataset_path = "/home/zrr/workspace/face-recognition/datasets"
    input_image = cv2.imread(f"{dataset_path}/Face-Dataset/UCEC-Face/subject1/subject1.4.png")
    return input_image
