from __future__ import annotations

import pickle
from itertools import product
from typing import Any, Sequence

import cv2
import opensmile
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def get_all_audio_methods(ignore_methods: Sequence[str] = ()):
    for feature_set, feature_level in product(opensmile.FeatureSet, opensmile.FeatureLevel):
        if feature_set.name in ignore_methods or feature_level.name in ignore_methods:
            continue
        if (
            feature_level == opensmile.FeatureLevel.LowLevelDescriptors_Deltas
            and feature_set != opensmile.FeatureSet.ComParE_2016
        ):
            continue

        match feature_set, feature_level:
            case opensmile.FeatureSet.ComParE_2016, opensmile.FeatureLevel.Functionals:
                feature_size = 6373
            case opensmile.FeatureSet.ComParE_2016, _:
                feature_size = 65
            case opensmile.FeatureSet.GeMAPS | opensmile.FeatureSet.GeMAPSv01b, opensmile.FeatureLevel.Functionals:
                feature_size = 62
            case (
                opensmile.FeatureSet.GeMAPS
                | opensmile.FeatureSet.GeMAPSv01b,
                opensmile.FeatureLevel.LowLevelDescriptors,
            ):
                feature_size = 18
            case (
                opensmile.FeatureSet.eGeMAPS
                | opensmile.FeatureSet.eGeMAPSv01b,
                opensmile.FeatureLevel.LowLevelDescriptors,
            ):
                feature_size = 23
            case (
                opensmile.FeatureSet.eGeMAPS
                | opensmile.FeatureSet.eGeMAPSv01b
                | opensmile.FeatureSet.eGeMAPSv02,
                opensmile.FeatureLevel.Functionals,
            ):
                feature_size = 88
            case opensmile.FeatureSet.eGeMAPSv02, opensmile.FeatureLevel.LowLevelDescriptors:
                feature_size = 25
            case opensmile.FeatureSet.emobase, opensmile.FeatureLevel.Functionals:
                feature_size = 988
            case opensmile.FeatureSet.emobase, opensmile.FeatureLevel.LowLevelDescriptors:
                feature_size = 26
            case _:
                raise ValueError(f"unknown feature_set: {feature_set}/{feature_level}")

        yield opensmile.Smile(feature_set=feature_set, feature_level=feature_level), feature_size
