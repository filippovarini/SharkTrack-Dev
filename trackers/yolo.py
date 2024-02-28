from ultralytics import YOLO
import time
from pathlib import Path
import sys
import os
from typing import List, Tuple

# Since we are importing a file in a super directory, we need to add the root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from trackers.utils import get_sorted_sequence

class YoloTracker:
  def __init__(self, model_path, tracker) -> None:
    self.model_path = model_path
    self.tracker = tracker

  def track(self, sequence_path, conf_threshold, iou_association_threshold, imgsz, tracker) -> Tuple[List, float]:
      mode = 'track' if tracker else 'detect'
      print(f'Running {mode} mode...')
      frames = get_sorted_sequence(sequence_path)


      model = YOLO(self.model_path)

      pred_bbox_xyxys = []
      pred_confidences = []
      pred_track_ids = []

      frame_count = 0

      sequence_start_time = time.time()
      for frame in frames:
          frame_count += 1
          print(f"\rProcessing frame {frame_count}", end='')

          frame_path = os.path.join(sequence_path, frame)

          if mode == 'track':
              results = model.track(
                  frame_path,
                  persist=True,
                  conf=conf_threshold,
                  iou=iou_association_threshold,
                  imgsz=imgsz,
                  tracker=tracker,
                  verbose=False
              )
              tracks = results[0].boxes.id
              track_ids = tracks.int().cpu().tolist() if tracks is not None else []
          else:  # mode == 'detect'
              results = model.predict(
                  frame_path,
                  conf=conf_threshold,
                  iou=iou_association_threshold,
                  imgsz=imgsz,
                  verbose=False
              )

          boxes = results[0].boxes.xyxy.cpu().tolist()
          confidences = results[0].boxes.conf.cpu().tolist()

          min_idx = min(len(boxes), len(confidences), len(track_ids) if mode == 'track' else len(boxes))

          pred_bbox_xyxys.append(boxes[:min_idx])
          pred_confidences.append(confidences[:min_idx])
          if mode == 'track':
              pred_track_ids.append(track_ids[:min_idx])
      
      track_time = time.time() - sequence_start_time

      print('\n')
      print(f'Processed {frame_count} frames in {mode} mode.')
      print(f'Sequence processing time: {track_time:.2f}s')

      if mode == 'track':
          assert len(pred_bbox_xyxys) == len(pred_confidences) == len(pred_track_ids), f'Lengths do not match {len(pred_bbox_xyxys)=}, {len(pred_confidences)=}, {len(pred_track_ids)=}'
      else:
          assert len(pred_bbox_xyxys) == len(pred_confidences), f'Lengths do not match {len(pred_bbox_xyxys)=}, {len(pred_confidences)=}'

      return [pred_bbox_xyxys, pred_confidences, pred_track_ids if mode == 'track' else []], track_time
