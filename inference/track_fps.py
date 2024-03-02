from ultralytics import YOLO
import cv2
import os
import time
import sys
from pathlib import Path

# Since we are importing a file in a super directory, we need to add the root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from annotations.inference import track_history_to_csv
from trackers.utils import get_sorted_sequence

def track_folder_videos(params):

  start_time = time.time()
  
  videos = os.listdir(params['video_folder'])
  for video in videos:
    print(f"Processing video {video}")

    video_start_time = time.time()

    video_path = os.path.join(params['video_folder'], video)
    assert os.path.exists(video_path), f'Video file does not exist {video_path}'
    cap = cv2.VideoCapture(video_path)

    # Calculate the frame skip based on the actual video fps and the desired fps
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = round(actual_fps / params['desired_fps'])

    model = YOLO(params['model_path'])

    track_history = {
      'pred_bbox_xyxys': [],
      'pred_confidences': [],
      'pred_track_ids': [],
    }

    i = 0
    frame_count = 0  # Initialize a counter for the frames processed

    # Loop through the video frames
    while cap.isOpened():
      # Read a frame from the video
      success, frame = cap.read()

      if not success:
        # finished reading
        break

      i += 1
      # Skip frames to process the video at the desired fps
      if i % frame_skip != 0:
        continue
      frame_count += 1

      print(f"\rProcessing frame {frame_count}", end='')

      # Run YOLOv8 tracking on the frame, persisting tracks between frames
      results = model.track(
        frame,
        persist=True,
        conf=params['conf_threshold'],
        iou=params['iou_association_threshold'],
        imgsz=params['imgsz'],
        tracker=params['tracker'],
        verbose=False
      )

      # Get the boxes and track IDs
      boxes = results[0].boxes.xyxy.cpu().tolist()
      tracks = results[0].boxes.id
      track_ids = tracks.int().cpu().tolist() if tracks is not None else []
      confidences = results[0].boxes.conf.cpu().tolist()

      min_idx = min(len(boxes), len(track_ids), len(confidences))

      # Store the track history
      track_history['pred_bbox_xyxys'].append(boxes[:min_idx])
      track_history['pred_confidences'].append(confidences[:min_idx])
      track_history['pred_track_ids'].append(track_ids[:min_idx])

    print('\n')
    print(f'processed {frame_count} frames')
    print(f'video processing time: {time.time() - video_start_time}')
    
    assert len(track_history['pred_bbox_xyxys']) == len(track_history['pred_confidences']) == len(track_history['pred_track_ids']), 'Lengths do not match'
    track_history_to_csv(track_history, video, params['annotation_folder'], params['desired_fps'])

    # Release the video capture object for the current video
    cap.release()

  print(f'Total time: {time.time() - start_time}')


if __name__ == '__main__':
  base_path = '/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/frame_extraction_raw'

  start_time = time.time()
  for folder in os.listdir(base_path):
      print(f'Processing folder {folder}...')
      video_folder = os.path.join(base_path, folder, 'videos')
      params = {
        'model_path': '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/models/p2v6_new_v4/weights/best.pt',
        'conf_threshold': 0.2,
        'iou_association_threshold': 0.5,
        'imgsz': 640,
        'tracker': '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/trackers/custom_botsort.yaml',
        'annotation_folder': f'/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/annotations/phase2_cstm_botsort_5fps/{folder}',
        'video_folder': video_folder,
        'desired_fps': 5,
      }
      track_folder_videos(params)

  print(f'Total time for all folders: {time.time() - start_time}')