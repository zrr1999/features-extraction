from __future__ import annotations

import cv2

from vision.utils import get_face_by_dlib, get_face_by_mediapipe


def float2int(value: float, max_value: int) -> int:
    return int(max_value * value)


dataset_path = "./datasets"
input_image = cv2.imread(f"{dataset_path}/Face-Dataset/UCEC-Face/subject1/subject1.4.png")
output_image = input_image.copy()


for get_face in [get_face_by_dlib, get_face_by_mediapipe]:
    face = get_face(input_image)
    if face:
        x, y, w, h = face
        cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imwrite("./outputs/detection.png", output_image)
