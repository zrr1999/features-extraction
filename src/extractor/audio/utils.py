from __future__ import annotations

from itertools import product
from typing import Sequence

import opensmile


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


def get_batch_size(feature_size,batch_size):
    if feature_size < 30:
        batch_size = batch_size
    if feature_size > 30 and feature_size < 90:
        batch_size = batch_size//2
    if feature_size == 988:
        batch_size = batch_size//4
    if feature_size == 6373:
        batch_size = batch_size//8
    return batch_size

# def get_all_audio_methods_if(ignore_methods: Sequence[str] = ()):
#     for feature_set, feature_level in product(opensmile.FeatureSet, opensmile.FeatureLevel):
#         if feature_set.name in ignore_methods or feature_level.name in ignore_methods:
#             continue
#         if (
#             feature_level == opensmile.FeatureLevel.LowLevelDescriptors_Deltas
#             and feature_set != opensmile.FeatureSet.ComParE_2016
#         ):
#             continue

#         if (
#             feature_set == opensmile.FeatureSet.ComParE_2016
#             and feature_level == opensmile.FeatureLevel.Functionals
#         ):
#             feature_size = 6373
#         elif feature_set == opensmile.FeatureSet.ComParE_2016:
#             feature_size = 65
#         elif (
#             feature_set in [opensmile.FeatureSet.GeMAPS, opensmile.FeatureSet.GeMAPSv01b]
#             and feature_level == opensmile.FeatureLevel.Functionals
#         ):
#             feature_size = 62
#         elif (
#             feature_set in [opensmile.FeatureSet.GeMAPS, opensmile.FeatureSet.GeMAPSv01b]
#             and feature_level == opensmile.FeatureLevel.LowLevelDescriptors
#         ):
#             feature_size = 18
#         elif (
#             feature_set in [opensmile.FeatureSet.eGeMAPS, opensmile.FeatureSet.eGeMAPSv01b]
#             and feature_level == opensmile.FeatureLevel.LowLevelDescriptors
#         ):
#             feature_size = 23
#         elif (
#             feature_set
#             in [opensmile.FeatureSet.eGeMAPS, opensmile.FeatureSet.eGeMAPSv01b, opensmile.FeatureSet.eGeMAPSv02]
#             and feature_level == opensmile.FeatureLevel.Functionals
#         ):
#             feature_size = 88
#         elif (
#             feature_set == opensmile.FeatureSet.eGeMAPSv02
#             and feature_level == opensmile.FeatureLevel.LowLevelDescriptors
#         ):
#             feature_size = 25
#         elif (
#             feature_set == opensmile.FeatureSet.emobase
#             and feature_level == opensmile.FeatureLevel.Functionals
#         ):
#             feature_size = 988
#         elif (
#             feature_set == opensmile.FeatureSet.emobase
#             and feature_level == opensmile.FeatureLevel.LowLevelDescriptors
#         ):
#             feature_size = 26
#         else:
#             raise ValueError(f"unknown feature_set: {feature_set}/{feature_level}")

#         yield opensmile.Smile(feature_set=feature_set, feature_level=feature_level), feature_size