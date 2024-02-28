import sys
import os
from pathlib import Path
from typing import List

# Since we are importing a file in a super directory, we need to add the root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from evaluation.utils import extract_frame_number

def get_sorted_sequence(sequence_path: str) -> List[str]:
  frames = [f for f in os.listdir(sequence_path) if f.endswith('.jpg')]
  frames.sort(key=extract_frame_number)
  return frames