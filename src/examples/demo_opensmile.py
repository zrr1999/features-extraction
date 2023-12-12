from __future__ import annotations

from pathlib import Path

import numpy as np
import opensmile

from extractor.audio.utils import get_all_audio_methods

dataset_path = "./datasets/ESD"
features_path = "./datasets/features/audio"


for smile, feature_size in get_all_audio_methods():
    y = smile.process_file(f"{dataset_path}/0001/Angry/train/0001_000700.wav")
    print(smile.feature_level, feature_size)
    print(f"{smile.feature_set}_{smile.feature_level}: {np.array(y).shape}")
