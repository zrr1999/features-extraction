from __future__ import annotations

import pickle

import numpy as np
import pandas as pd


def emotion2int(emotion: str):
    emotion2int = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "fear": 4, "disgust": 5, "surprise": 6}
    return emotion2int[emotion]


meta = pd.read_csv(
    "/home/zrr/workspace/emotion-reproduce/datasets/MELD.Raw/train_sent_emo.csv", sep=",", index_col=0, header=0
)

mapping_method_path = {
    "r21d": "r21d/r21d/r2plus1d_18_16_kinetics",
    "s3d": "s3d/s3d",
    # "resnet": "resnet/resnet/resnet50"
}

for method, method_path in mapping_method_path.items():
    all_features = {}
    for index, row in meta.iterrows():
        utt_id = row["Utterance_ID"]
        dia_id = row["Dialogue_ID"]
        path = f"/home/zrr/workspace/emotion-reproduce/features/train_splits/{method_path}/dia{dia_id}_utt{utt_id}_{method}.npy"
        try:
            features = np.load(path)
            if features.shape[0] != 0:
                all_features[(emotion2int(row["Emotion"]), index)] = features
        except FileNotFoundError:
            print(f"FileNotFoundError: {path}")

    with open(f"datasets/features/video/{method}_features.pkl", "wb") as file:
        pickle.dump(all_features, file)
    print(method)
