from ultralytics import YOLO
import cv2
import os
import time
import sys
from pathlib import Path
import pandas as pd

# Since we are importing a file in a super directory, we need to add the root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from annotations.inference import track_history_to_csv

def track_folder_videos(params):

  assert os.path.exists(params['video_path']), f'Video file does not exist {params["video_path"]}'  
  cap = cv2.VideoCapture(params['video_path'])  

  # Calculate the frame skip based on the actual video fps and the desired fps
  actual_fps = cap.get(cv2.CAP_PROP_FPS)
  frame_skip = round(actual_fps / params['fps'])

  model = YOLO(params['model_path'])

  if params['tracked']:
    prediction_start_time = time.time()
    results = model.track(
      params['video_path'],
      conf=params['conf_threshold'],
      iou=params['iou_association_threshold'],
      imgsz=params['imgsz'],
      tracker=params['tracker'],
      vid_stride=frame_skip,
      device=params['device'],
      verbose=False,
    )
    prediction_end_time = time.time()
  else:
    prediction_start_time = time.time()
    results = model.predict(
      params['video_path'],
      conf=params['conf_threshold'],
      iou=params['iou_association_threshold'],
      imgsz=params['imgsz'],
      tracker=params['tracker'],
      vid_stride=frame_skip,
      device=params['device'],
      verbose=False,
    )
    prediction_end_time = time.time()
    
  return prediction_end_time - prediction_start_time


def save_to_csv(params):
  csv_path = '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/tests/model_speeds.csv'
  columns = ['device', 'fps', 'size', 'tracked', 'imgsz', 'time']

  # Create a new dictionary from params with only the keys we want
  new_params = {key: params[key] for key in columns}

  df = pd.read_csv(csv_path)
  new_row = pd.DataFrame(new_params, index=[0])
  df = pd.concat([df, new_row], ignore_index=True)
  df.to_csv(csv_path, index=False)


def main():
  video_to_test = '/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/frame_extraction_raw/sp/videos/sp_palau4.mp4'

  device = ['cpu', 'cuda']
  fps = [0.5, 1, 2, 5]
  size = ['n', 's', 'm']
  tracked = [True, False]
  img_size = [640, 1280]

  model_paths = {
    'm': '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/models/p2v5_new/weights/best.pt',
    's': '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/models/yolov8_s_mvd4_50/weights/best.pt',
    'n': '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/models/yolov8_n_mvd2_50/best.pt'
  }


  for d in device:
    for f in fps:
      for s in size:
        for t in tracked:
          for i in img_size:
            print(f"Testing device: {d}, fps: {f}, size: {s}, tracked: {t}, imgsz: {i}")
            params = {
              'device': d,
              'size': s,
              'tracked': t,
              'imgsz': i,
              'video_path': video_to_test,
              'model_path': model_paths[s],
              'conf_threshold': 0.2,
              'iou_association_threshold': 0.5,
              'tracker': 'botsort.yaml',
              'fps': f,
            }
            time_taken = track_folder_videos(params)
            params['time'] = time_taken
            save_to_csv(params)

if __name__ == '__main__':
  main()
