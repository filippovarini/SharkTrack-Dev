from ultralytics import YOLO
import cv2
import pandas as pd
import numpy as np
import os
import time 
from pathlib import Path
import sys

# Since we are importing a file in a super directory, we need to add the root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from utils import align_annotations_with_predictions_dict_corrected, evaluate_tracking, get_torch_device

BRUVS_VIDEO_LENGTH = 20
VAL_VIDEOS = [
  'val1_medium1',
  'gfp_bahamas1',
  'gfp_palau1',
  'shlife_scalloped2',
  'gfp_polynesia1',
  'shlife_bull6',
  'gfp_maldives1',
  'gfp_rand5',
  'gfp_barbados1',
  'gfp_solomon1',
  'val1_easy2',
  'gfp_kiribati1'
]
DATASET_PATH = '/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/phase2'

def get_frames_sequences(frames_path):
  frames_sequence = [f for f in os.listdir(frames_path) if f.endswith('.jpg')]
  frames_sequence.sort(key=lambda x: int(x.split('_')[-1].split('.')[0].replace('frame', '')))
  return frames_sequence


def track(model_path, frames_folder, conf, iou, imgsz, tracker):
  """
  frames: sorted list of frames
  """
  assert tracker in ['botsort.yaml', 'bytetrack.yaml']

  model = YOLO(model_path)

  pred_bboxes = []
  pred_confidences = []
  pred_track_ids = []

  frames = get_frames_sequences(frames_folder)

  i = 0
  for frame_name in frames:
    print(f"\rTracking frame {frame_name}", end='')

    frame_path = os.path.join(frames_folder, frame_name)
    frame = cv2.imread(frame_path)

    i += 1
    results = model.track(
      frame,
      persist=True,
      conf=conf,
      iou=iou,
      imgsz=imgsz,
      tracker=str(tracker),
      verbose=False
    )

    # Get the boxes and track IDs
    boxes = results[0].boxes.xywh.cpu().tolist()
    tracks = results[0].boxes.id
    confidences = results[0].boxes.conf.cpu().tolist()
    track_ids = tracks.int().cpu().tolist() if tracks is not None else []
    assert len(boxes) == len(confidences), f'bboxes and confidence must be same length, resp: {len(boxes)}, {len(confidences)}'
    pred_idx = min(len(boxes), len(track_ids))

    pred_bboxes.append(boxes[:pred_idx])
    pred_confidences.append(confidences[:pred_idx])
    pred_track_ids.append(track_ids[:pred_idx])

  return [pred_bboxes, pred_confidences, pred_track_ids]
  

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

 
def evaluate(model_path, conf, iou, imgsz, tracker, project_path):
  """
  1. Evaluate object detection model using the evaluation dataset
  2. Evaluate object detection model using the out-of-distribution evaluation dataset
  3. Evaluate tracker model using the evaluation dataset
  """
  # macro average
  motas = []
  motps = []
  idf1s = []

  # Prepare performance plot
  # num_plots = len(video_names)
  # performance_plot, axs = plt.subplots(num_plots, 1, figsize=(10, 6 * num_plots))

  track_start_time = time.time()

  i = 0
  
  sequences = os.listdir(DATASET_PATH)
  for sequence in sequences:
    if i ==1:
      break
    i+=1
    frames_folder = os.path.join(DATASET_PATH, sequence)
    annotations_path = os.path.join(frames_folder, 'annotations.csv')
    annotations = pd.read_csv(annotations_path)
    
    print(f'Evaluating {sequence}...')
    
    results = track(model_path, frames_folder, conf, iou, imgsz, tracker)

    # Extract and store annotations for investigation
    # extracted_pred_results = extract_tracks(results)
    extracted_pred_results = results
    aligned_annotations = align_annotations_with_predictions_dict_corrected(annotations, extracted_pred_results, BRUVS_VIDEO_LENGTH)
    aligned_annotations['frame_id'] = [i for i in range(len(aligned_annotations['gt_bbox_xyxys']))]
    aligned_annotations_df = pd.DataFrame(aligned_annotations)
    aligned_annotations_path = os.path.join(project_path, 'annotatinos.csv')
    aligned_annotations_df.to_csv(aligned_annotations_path, index=False)

    mota, motp, idf1, frame_avg_motp = evaluate_tracking(aligned_annotations, S_TRESH=0.5)
    print(f'{sequence} - MOTA: {round(mota, 2)}, MOTP: {round(motp, 2)}, IDF1: {round(idf1, 2)}')
    motas.append(mota)
    motps.append(motp)
    idf1s.append(idf1)

    # fig = utils.plot_performance_graph(aligned_annotations, frame_avg_motp)
    # fig.savefig(os.path.join(model_folder, f"{video}.png"))

  macro_mota = round(np.mean(motas), 2)
  macro_motp = round(np.mean(motps), 2)
  macro_idf1 = round(np.mean(idf1s), 2)

  track_end_time = time.time()
  track_time = round((track_end_time - track_start_time) / 60, 2)

  return macro_mota, macro_motp, macro_idf1, track_time, get_torch_device()#, performance_plot


# test
evaluate('/vol/biomedic3/bglocker/ugproj2324/fv220/dev/old/shark_locator_tests/runs/detect/yolov8m_mvd2/best.pt', 0.2, 0.5, 640, 'botsort.yaml', '.')