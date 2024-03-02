import sys
sys.path.append('/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev')

from evaluation.evaluate_yolo_tracker import evaluate_sequence, VAL_SEQUENCES
import pandas as pd
import yaml
import os


IOU_ASSOCIATION_TRESHOLD = 0.5
CONF_TRESHOLD = 0.2
project = '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/models/p2v4_new_1000e_no_patience/'
custom_botsort_path = '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/trackers/custom_botsort5fps.yaml'

# Hyperparameter tuning
# tracker_types = ['botsort', 'bytesort']
tracker_types = ['botsort']
track_high_threshs = [0.4, 0.6, 0.8]
track_low_threshs = [0.1, 0.2]
new_track_threshs = [0.4, 0.6, 0.8]
track_buffers = [3, 5, 10, 15] # 1 per second = buffer 2
match_threshs = [0.8, 0.9, 0.95, 0.97, 0.98]

df_path = '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/SharkTrack-Dev/evaluation/hyperparameter_tuning/5fps.csv'

# if not already present, create empty dataframe with columns
if not os.path.exists(df_path):
  df = pd.DataFrame(columns=['tracker_type', 'track_high_thresh', 'track_low_thresh', 'new_track_thresh', 'track_buffer', 'match_thresh', 'mota', 'motp', 'idf1', 'hota'])
  df.to_csv(df_path, index=False)

def tune():
  for tracker_type in tracker_types:
    for track_high_thresh in track_high_threshs:
      for track_low_thresh in track_low_threshs:
        for new_track_thresh in new_track_threshs:
          for track_buffer in track_buffers:
            for match_thresh in match_threshs:
              if track_low_thresh >= track_high_thresh:
                continue

              botsort_settings = {
                'tracker_type': tracker_type,
                'track_high_thresh': track_high_thresh,
                'track_low_thresh': track_low_thresh,
                'new_track_thresh': new_track_thresh,
                'track_buffer': track_buffer,
                'match_thresh': match_thresh,
                'gmc_method': 'sparseOptFlow',
                'proximity_thresh': 0.2,
                'appearance_thresh': 0.25,
                'with_reid': False,
              }

              print(botsort_settings)

              # Writing to a YAML file
              with open(custom_botsort_path, 'w') as file:
                documents = yaml.dump(botsort_settings, file)
              
              model_path = os.path.join(project, 'weights/best.pt')
              motas, motps, idf1s, hotas, track_time, _, aligned_annotations_list = evaluate_sequence(model_path, CONF_TRESHOLD, IOU_ASSOCIATION_TRESHOLD, imgsz=640, tracker_type=custom_botsort_path)

              # append row to the dataframe and save it
              df = pd.read_csv(df_path)
              new_row = pd.DataFrame({
                'tracker_type': [tracker_type],
                'track_high_thresh': [track_high_thresh],
                'track_low_thresh': [track_low_thresh],
                'new_track_thresh': [new_track_thresh],
                'track_buffer': [track_buffer],
                'match_thresh': [match_thresh],
                'mota': [motas],
                'motp': [motps],
                'idf1': [idf1s],
                'hota': [hotas]
              })
              df = pd.concat([df, new_row], ignore_index=True)
              df.to_csv(df_path, index=False)


tune()