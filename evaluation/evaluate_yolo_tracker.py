from utils import align_annotations_with_predictions_dict_corrected, evaluate_tracking, get_torch_device
from ultralytics import YOLO
import pandas as pd
import numpy as np
import os
import time 

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


def get_frame_sequences(frames_folder):
  frame_sequences = {}
  annotations = {}

  for video in VAL_VIDEOS:
    sequence = [f for f in os.path.listdir(os.path.join(frames_folder, video)) if f.endswith('.jpg')]
    sequence.sort(key=lambda x: int(x.split('_')[-1].split('.')[0].replace('frame', '')))
    frame_sequences[video] = sequence
    annotations[video] = os.path.join(frames_folder, video, 'annotations.csv')
  
  return frame_sequences, annotations


 
def evaluate(model_path, conf, iou, imgsz, tracker, project_path):
  """
  1. Evaluate object detection model using the evaluation dataset
  2. Evaluate object detection model using the out-of-distribution evaluation dataset
  3. Evaluate tracker model using the evaluation dataset
  """
  frames_folder = '/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/phase2/'
  
  frame_sequences, annotations = get_frame_sequences(frames_folder)
  assert sorted(frame_sequences.keys()) == sorted(annotations.keys()), f'frame sequences and annotations must be the same {frame_sequences}, {annotations}'

  # macro average
  motas = []
  motps = []
  idf1s = []

  # Prepare performance plot
  # num_plots = len(video_names)
  # performance_plot, axs = plt.subplots(num_plots, 1, figsize=(10, 6 * num_plots))

  track_start_time = time.time()
  
  for video in frame_sequences:
    annotations_path = annotations[video]
    annotations = pd.read_csv(annotations_path)
    
    print(f'Evaluating {video}...')
    
    results = track(model_path, frame_sequences[video], conf, iou, imgsz, tracker)

    # Extract and store annotations for investigation
    extracted_pred_results = extract_tracks(results)
    aligned_annotations = align_annotations_with_predictions_dict_corrected(annotations, extracted_pred_results, BRUVS_VIDEO_LENGTH)
    aligned_annotations['frame_id'] = [i for i in range(len(aligned_annotations['gt_bbox_xyxys']))]
    aligned_annotations_df = pd.DataFrame(aligned_annotations)
    aligned_annotations_path = os.path.join(project_path, 'annotatinos.csv')
    aligned_annotations_df.to_csv(aligned_annotations_path, index=False)

    mota, motp, idf1, frame_avg_motp = evaluate_tracking(aligned_annotations, S_TRESH=0.5)
    print(f'{video} - MOTA: {round(mota, 2)}, MOTP: {round(motp, 2)}, IDF1: {round(idf1, 2)}')
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