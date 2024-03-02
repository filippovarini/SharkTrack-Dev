from ultralytics import YOLO
import pandas as pd
import numpy as np
import time 
import os
from pathlib import Path
import sys

# Since we are importing a file in a super directory, we need to add the root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from trackers.yolo import YoloTracker
from trackers.sort_adapter import Sort_adapter
from evaluation.utils import target2pred_align, get_torch_device, plot_performance_graph, extract_frame_number, save_trackeval_annotations
from evaluation.TrackEval.scripts.run_mot_challenge_functional import run_mot_challenge


sequences_path = '/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/phase2'
sequences_path = '/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/frame_extraction_raw/val1/frames_5fps'
VAL_SEQUENCES = [
  'val1_difficult1',
  'val1_difficult2',
  'val1_easy1',
  'val1_easy2',
  'val1_medium1',
  'val1_medium2',
  # 'sp_natgeo2',
  # 'gfp_hawaii1',
  # 'shlife_scalloped4',
  # 'gfp_fiji1',
  # 'shlife_smooth2',
  # 'gfp_niue1',
  # 'gfp_solomon1',
  # 'gfp_montserrat1',
  # 'gfp_rand3',
  # 'shlife_bull4'
]
tracker_class = {
  'botsort': YoloTracker,
  'bytetrack': YoloTracker,
  'sort': Sort_adapter
}



def compute_clear_metrics():
  sequence_metrics = run_mot_challenge(BENCHMARK='val1', METRICS=['CLEAR', 'Identity', 'HOTA'])
  motas, motps, idf1s, hotas = 0, 0, 0, 0
  for sequence in sequence_metrics:
    mota = round(sequence_metrics[sequence]['MOTA'], 2)
    motp = round(sequence_metrics[sequence]['MOTP'], 2)
    idf1 = round(sequence_metrics[sequence]['IDF1'], 2)
    hota = round(sequence_metrics[sequence]['HOTA(0)'], 2)
    motas += mota
    motps += motp
    idf1s += idf1
    hotas += hota
  
  motas = round(motas / len(sequence_metrics), 2)
  motps = round(motps / len(sequence_metrics), 2)
  idf1s = round(idf1s / len(sequence_metrics), 2)
  hotas = round(hotas / len(sequence_metrics), 2)

  return motas, motps, idf1s, hotas

def evaluate_sequence(model_path, conf_threshold, iou_association_threshold, imgsz, tracker_type):
  all_aligned_annotations = {}
  track_time = 0
  for sequence in VAL_SEQUENCES:
    sequence_path = os.path.join(sequences_path, sequence)
    assert os.path.exists(sequence_path), f'sequence file does not exist {sequence_path}'

    annotations_path = os.path.join(sequence_path, 'annotations.csv')
    assert os.path.exists(annotations_path), f'annotations file does not exist {annotations_path}'
    annotations = pd.read_csv(annotations_path)

    print(f"Evaluating {sequence}")
    tracker_obj = tracker_class.get(tracker_type, YoloTracker) # default to YoloTracker for custom trackers
    tracker = tracker_obj(model_path, tracker_type)
    results, time = tracker.track(sequence_path, conf_threshold, iou_association_threshold, imgsz) # [bbox_xyxys, confidences, track_ids]
    track_time += time

    # Annotations for visualisation
    aligned_annotations = target2pred_align(annotations, results, sequence_path, tracker=tracker_type)
    all_aligned_annotations[sequence] = (aligned_annotations)

    # plot_performance_graph(aligned_annotations, sequence)

  motas, motps, idf1s = 0, 0, 0 
  
  if tracker:
    # save prediction annotations to calculate metrics
    save_trackeval_annotations(all_aligned_annotations)
    motas, motps, idf1s, hotas = compute_clear_metrics()

  return motas, motps, idf1s, hotas, track_time, get_torch_device(), all_aligned_annotations

def evaluate(model_path, conf, iou, imgsz, tracker):
  """
  return macro-avg metrics
  """
  motas, motps, idf1s, hotas, track_time, device, _ = evaluate_sequence(model_path, conf, iou, imgsz, tracker)

  macro_mota = round(np.mean(motas), 2)
  macro_motp = round(np.mean(motps), 2)
  macro_idf1 = round(np.mean(idf1s), 2)

  return macro_mota, macro_motp, macro_idf1, track_time, device

def track(model_path, video_path, conf, iou, imgsz, tracker):
  assert tracker in ['botsort.yaml', 'bytetrack.yaml']

  model = YOLO(model_path)

  results = model.track(
    source=video_path,
    persist=True,
    conf=conf,
    iou=iou,
    imgsz=imgsz,
    tracker=str(tracker),
    verbose=False
  )

  return results

def extract_tracks(results):
  """
  Convert Yolo-style results to list of tracks to be matched with annotation format
  :param results: List of predictions in the format results.bbox = [bbox_xyxy], [confidences], [track_ids]
  """
  bbox_xyxys = []
  confidences = []
  track_ids = []

  for i in range(len(results)):
    bbox_xyxy = []
    confidence = []
    track_id = []
    if results[i].boxes.id is not None:
      bbox_xyxy = results[i].boxes.xyxy.tolist()
      confidence = results[i].boxes.conf.tolist()
      track_id = results[i].boxes.id.tolist()

    bbox_xyxys.append(bbox_xyxy)
    confidences.append(confidence)
    track_ids.append(track_id)

  return [bbox_xyxys, confidences, track_ids]

 
