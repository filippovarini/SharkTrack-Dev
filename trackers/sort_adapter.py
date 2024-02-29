from ultralytics import YOLO
import time
from pathlib import Path
import sys
import os
import numpy as np
from typing import List, Tuple

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from trackers.utils import get_sorted_sequence
from trackers.sort import Sort

class Sort_adapter:
  def __init__(self, model_path, tracker_type) -> None:
    self.model_path = model_path
    self.tracker_type = tracker_type
    # Hyperparameters
    self.max_age = 1
    self.min_hits = 1
    self.iou_threshold = 0.8

  def track(self, sequence_path, conf_threshold, iou_association_threshold, imgsz) -> Tuple[List, float]:
    """
    Returns: [pred_bbox_xyxys, pred_confidences, pred_track_ids if mode == 'track' else []], track_time
    each of the lists in the returned list has length equal to the number of frames in the sequence and 
    contains the predicted bounding boxes, confidences, and track ids (if mode == 'track') for each frame
    """
    print(f'Running {self.tracker_type} tracker mode...')
    frames = get_sorted_sequence(sequence_path)

    model = YOLO(self.model_path)
    tracker = Sort(
        max_age=self.max_age,
        min_hits=self.min_hits,
        iou_threshold=self.iou_threshold
    )

    pred_bbox_xyxys = []
    pred_confidences = []
    pred_track_ids = []

    frame_count = 0

    sequence_start_time = time.time()
    for frame in frames:
        frame_count += 1
        print(f"\rProcessing frame {frame_count}", end='')

        frame_path = os.path.join(sequence_path, frame)

        results = model.predict(
            frame_path,
            conf=conf_threshold,
            iou=iou_association_threshold,
            imgsz=imgsz,
            verbose=False
        )

        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        confidences = results[0].boxes.conf.cpu().numpy().tolist()
        assert len(boxes) == len(confidences), f'Lengths do not match {len(boxes)=}, {len(confidences)=}'
        detections = np.column_stack((boxes, confidences)) if len(boxes) > 0 else np.empty((0, 5))
        trackers = tracker.update(detections)

        pred_bbox_xyxys.append(trackers[:, :4].tolist())
        pred_confidences.append(np.ones(trackers.shape[0]).tolist()) # TODO: get real confidence
        pred_track_ids.append(trackers[:, 4].tolist())
    
    track_time = time.time() - sequence_start_time

    print('\n')
    print(f'Processed {frame_count} frames in track mode.')
    print(f'Sequence processing time: {track_time:.2f}s')

    assert len(pred_bbox_xyxys) == len(pred_confidences), f'Lengths do not match {len(pred_bbox_xyxys)=}, {len(pred_confidences)=}'

    return [pred_bbox_xyxys, pred_confidences, pred_track_ids], track_time
