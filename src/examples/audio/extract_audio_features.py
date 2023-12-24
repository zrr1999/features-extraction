from __future__ import annotations

import itertools
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import opensmile
from loguru import logger
from rich.progress import Progress

from extractor.audio.utils import get_all_audio_methods
from extractor.vision.utils import detect_face, extract_face_features, get_all_face_methods

dataset_path = "./datasets/ESD"
features_path = "./datasets/features/audio"

os.makedirs(features_path, exist_ok=True)

for (smile, _), set_type in itertools.product(get_all_audio_methods(), ["train", "evaluation", "test"]):
    emotions = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
    data = {}
    with Progress() as progress:
        task = progress.add_task(f"[red]Generating pkl", total=100)
        for e, i in itertools.product(emotions, range(1, 21)):
            progress.update(task, advance=1)
            sub_dir = Path(f"{dataset_path}/{i:04d}")
            emotion_dir = sub_dir / e / set_type
            for j, f in enumerate(emotion_dir.iterdir()):
                data[(emotions.index(e), i + j * 20)] = np.array(smile.process_file(f.as_posix()))
                print(f.as_posix(), data[(emotions.index(e), i + j * 20)].shape)

    with open(f"{features_path}/{set_type}_{smile.feature_set}_{smile.feature_level}.pkl", "wb") as file:
        pickle.dump(data, file)
