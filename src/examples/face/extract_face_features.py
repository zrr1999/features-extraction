from __future__ import annotations

import itertools
import os
import pickle

import cv2
from loguru import logger
from rich.progress import Progress

from extractor.vision.utils import detect_face, extract_face_features, get_all_face_methods

dataset_path = "./datasets/Face-Dataset/UCEC-Face"
features_path = "./datasets/features/vision"

os.makedirs(features_path, exist_ok=True)

for detection_method, recognition_method, _ in get_all_face_methods():
    if os.path.exists(f"{features_path}/{detection_method}_{recognition_method}.pkl"):
        logger.info(f"skip: {features_path}/{detection_method}_{recognition_method}.pkl exists")
        continue

    data = {}

    with Progress() as progress:
        task1 = progress.add_task(f"[red]Using {detection_method} and {recognition_method} to generate pkl", total=131)
        for i in range(1, 131):
            progress.update(task1, advance=1)
            subject_path = f"{dataset_path}/subject{i}"
            for j, image_name in enumerate(os.listdir(subject_path)):
                image_path = f"{subject_path}/{image_name}"
                input_image = cv2.imread(image_path)
                output_image = input_image.copy()

                face = detect_face(input_image, detector=detection_method)

                if face:
                    x, y, w, h = face
                    face_image = input_image[y : y + h, x : x + w]
                    features = extract_face_features(face_image, model_name=recognition_method)
                    data[(i, j)] = (image_path, features)
                else:
                    logger.warning(f"no face detected: {image_path}")

    with open(f"{features_path}/{detection_method}_{recognition_method}.pkl", "wb") as file:
        pickle.dump(data, file)

    with open(f"{features_path}/{detection_method}_{recognition_method}.pkl", "rb") as file:
        loaded_data = pickle.load(file)
        assert data == loaded_data
