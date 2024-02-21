from utils import target2pred_align, evaluate_tracking, get_torch_device, plot_performance_graph, extract_frame_number
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
  'val1_easy1',
  'val1_easy2',
  'val1_medium1',
  'val1_medium2',
  'sp_natgeo2',
  'gfp_hawaii1',
  'shlife_scalloped4',
  'gfp_fiji1',
  'shlife_smooth2',
  'gfp_niue1',
  'gfp_solomon1',
  'gfp_montserrat1',
  'gfp_rand3',
  'shlife_bull4'
]


def track_frame_sequence(sequence_path, model_path, conf_threshold, iou_association_threshold, imgsz, tracker):
    frames = [f for f in os.listdir(sequence_path) if f.endswith('.jpg')]
    frames.sort(key=extract_frame_number)

    sequence_start_time = time.time()

    model = YOLO(model_path)

    bbox_xyxys = []
    confidences = []
    track_ids = []

    frame_count = 0 

    for frame in frames:
      frame_count += 1
      frame_number = extract_frame_number(frame)
      print(f"Processing frame {frame_number}")

      results = model.track(
        frame,
        persist=True,
        conf=conf_threshold,
        iou=iou_association_threshold,
        imgsz=imgsz,
        tracker=tracker,
        verbose=False
      )

      # Get the boxes and track IDs
      boxes = results[0].boxes.xyxy.cpu().tolist()
      tracks = results[0].boxes.id
      track_ids = tracks.int().cpu().tolist() if tracks is not None else []
      confidences = results[0].boxes.conf.cpu().tolist()

      min_idx = min(len(boxes), len(track_ids), len(confidences))

      # Store the track history
      bbox_xyxys.append(boxes[:min_idx])
      confidences.append(confidences[:min_idx])
      track_ids.append(track_ids[:min_idx])

    print('\n')
    print(f'processed {frame_count} frames')
    print(f'sequence processing time: {time.time() - sequence_start_time}')
    assert len(bbox_xyxys) == len(confidences) == len(track_ids), f'Lengths do not match {len(bbox_xyxys)=}, {len(confidences)=}, {len(track_ids)=}'

    return [bbox_xyxys, confidences, track_ids]


def evaluate_sequence(model_path, conf_threshold, iou_association_threshold, imgsz, tracker):

  # macro average
  motas = []
  motps = []
  idf1s = []
  figs = []

  # Prepare performance plot
  num_plots = len(VAL_SEQUENCES)
  performance_plot, axs = plt.subplots(num_plots, 1, figsize=(10, 6 * num_plots))

  track_start_time = time.time()
  for sequence in VAL_SEQUENCES:
    sequence_path = os.path.join(sequences_path, sequence)
    assert os.path.exists(sequence_path), f'sequence file does not exist {sequence_path}'

    annotations_path = os.path.join(sequence_path, 'annotations.csv')
    assert os.path.exists(annotations_path), f'annotations file does not exist {annotations_path}'
    annotations = pd.read_csv(annotations_path)

    print(f"Evaluating {sequence}")
    results = track_frame_sequence(sequence_path, model_path, conf_threshold, iou_association_threshold, imgsz, tracker)

    aligned_annotations = target2pred_align(annotations, results, sequence_path)
    aligned_annotations_df = pd.DataFrame(aligned_annotations)
    # save annotations in future

    mota, motp, idf1, frame_avg_motp = evaluate_tracking(aligned_annotations, S_TRESH=0.5)
    motas.append(mota)
    motps.append(motp)
    idf1s.append(idf1)

    fig = plot_performance_graph(aligned_annotations, frame_avg_motp)
    figs.append(fig)
  
  track_end_time = time.time()
  track_time = round((track_end_time - track_start_time) / 60, 2)

  return motas, motps, idf1s, track_time, get_torch_device(), figs


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

 
