from utils import target2pred_align, evaluate_tracking, get_torch_device, plot_performance_graph, extract_frame_number, save_trackeval_annotations
from TrackEval.scripts.run_mot_challenge_functional import run_mot_challenge
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
import numpy as np
import time 
import cv2
import os


BRUVS_VIDEO_LENGTH = 20
sequences_path = '/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/phase2'
VAL_SEQUENCES = [
  'val1_difficult1',
  'val1_difficult2',
  # 'val1_easy1',
  # 'val1_easy2',
  # 'val1_medium1',
  # 'val1_medium2',
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

def compute_clear_metrics():
  sequence_metrics = run_mot_challenge(BENCHMARK='val1', METRICS=['CLEAR', 'Identity'])
  motas, motps, idf1s = 0, 0, 0
  for sequence in sequence_metrics:
    mota = round(sequence_metrics[sequence]['MOTA'], 2)
    motp = round(sequence_metrics[sequence]['MOTP'], 2)
    idf1 = round(sequence_metrics[sequence]['IDF1'], 2)
    print(f'{sequence} MOTA: {mota}, MOTP: {motp}, IDF1: {idf1}')
    motas += mota
    motps += motp
    idf1s += idf1
  
  motas = round(motas / len(sequence_metrics), 2)
  motps = round(motps / len(sequence_metrics), 2)
  idf1s = round(idf1s / len(sequence_metrics), 2)

  return motas, motps, idf1s

def process_frame_sequence(sequence_path, model_path, conf_threshold, iou_association_threshold, imgsz, tracker=None):
    mode = 'track' if tracker else 'detect'
    print(f'Running {mode} mode...')
    frames = [f for f in os.listdir(sequence_path) if f.endswith('.jpg')]
    frames.sort(key=extract_frame_number)

    sequence_start_time = time.time()

    model = YOLO(model_path)

    pred_bbox_xyxys = []
    pred_confidences = []
    pred_track_ids = []

    frame_count = 0

    for frame in frames:
        frame_count += 1
        frame_number = extract_frame_number(frame)
        print(f"\rProcessing frame {frame_number}", end='')

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

    print('\n')
    print(f'Processed {frame_count} frames in {mode} mode.')
    print(f'Sequence processing time: {time.time() - sequence_start_time}')

    if mode == 'track':
        assert len(pred_bbox_xyxys) == len(pred_confidences) == len(pred_track_ids), f'Lengths do not match {len(pred_bbox_xyxys)=}, {len(pred_confidences)=}, {len(pred_track_ids)=}'
    else:
        assert len(pred_bbox_xyxys) == len(pred_confidences), f'Lengths do not match {len(pred_bbox_xyxys)=}, {len(pred_confidences)=}'

    return [pred_bbox_xyxys, pred_confidences, pred_track_ids if mode == 'track' else []]

def evaluate_sequence(model_path, conf_threshold, iou_association_threshold, imgsz, tracker):
  all_aligned_annotations = {}
  track_time = 0
  for sequence in VAL_SEQUENCES[:5]:
    sequence_path = os.path.join(sequences_path, sequence)
    assert os.path.exists(sequence_path), f'sequence file does not exist {sequence_path}'

    annotations_path = os.path.join(sequence_path, 'annotations.csv')
    assert os.path.exists(annotations_path), f'annotations file does not exist {annotations_path}'
    annotations = pd.read_csv(annotations_path)

    print(f"Evaluating {sequence}")
    results = process_frame_sequence(sequence_path, model_path, conf_threshold, iou_association_threshold, imgsz, tracker) # [bbox_xyxys, confidences, track_ids]

    # Annotations for visualisation
    aligned_annotations = target2pred_align(annotations, results, sequence_path, tracker=tracker)
    all_aligned_annotations[sequence] = (aligned_annotations)

    plot_performance_graph(aligned_annotations, sequence)

  motas, motps, idf1s = 0, 0, 0 
  
  if tracker:
    # save prediction annotations to calculate metrics
    save_trackeval_annotations(all_aligned_annotations)
    motas, motps, idf1s = compute_clear_metrics()
  track_time = 0

  return motas, motps, idf1s, track_time, get_torch_device(), all_aligned_annotations

def evaluate(model_path, conf, iou, imgsz, tracker):
  """
  return macro-avg metrics
  """
  motas, motps, idf1s, track_time, device, _ = evaluate_sequence(model_path, conf, iou, imgsz, tracker)

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

 
