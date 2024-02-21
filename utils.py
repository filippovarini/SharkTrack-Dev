import matplotlib.pyplot as plt
import pandas as pd
import torch
import os


def get_torch_device():
    """
    Returns the device to be used for training and inference.
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    else:
        return "CPU"