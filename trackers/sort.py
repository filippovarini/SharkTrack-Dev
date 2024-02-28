from ultralytics import YOLO
import time
from pathlib import Path
import sys
import os
from typing import List, Tuple

class SORT:
  def __init__(self, model_path) -> None:
    self.trackers = []

  def track() -