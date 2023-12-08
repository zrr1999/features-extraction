from __future__ import annotations

import pickle
from itertools import product
from typing import Any, Sequence

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def get_all_audio_methods(ignore_methods: Sequence[str] = ()):
    pass
