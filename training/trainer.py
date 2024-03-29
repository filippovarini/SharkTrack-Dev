from ultralytics import YOLO
import pandas as pd
import wandb
import os
import sys
from pathlib import Path

# Since we are importing a file in a super directory, we need to add the root directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from evaluation.evaluate_yolo_tracker import evaluate

params = {
  'name': 'p2v6_new_v4',
  'model_size': 'x',
  'pretrained_model': None,
  # 'pretrained_model': '/vol/biomedic3/bglocker/ugproj2324/fv220/dev/old/shark_locator_tests/runs/detect/yolov8m_mvd2/best.pt',
  'epochs': 1000,
  'imgsz': 640,
  'patience': 200,
  'data_yaml': '/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/experimentation_datasets/p2v6/data_config.yaml',
  'project_folder': 'models',
  'batch': 64,

  "iou_association_threshold": 0.5,
  "tracker": "botsort.yaml",
  "conf_threshold": 0.2,
}

model = YOLO(params['pretrained_model'] or f"yolov8{params['model_size']}.pt")  # load a pretrained model (recommended for training)

# Train the model
model.train(
  data=params['data_yaml'],
  epochs=params['epochs'],
  imgsz=params['imgsz'],
  patience=params['patience'],
  name=params['name'],
  batch=params['batch'],
  project = params['project_folder'],
  verbose=True,
  save_period=50
)


# Get mAP
model_folder = os.path.join(params['project_folder'], params['name'])
assert os.path.exists(model_folder), 'Model folder does not exist'
results_path = os.path.join(model_folder, 'results.csv')
assert os.path.exists(results_path), 'Results file does not exist'
results = pd.read_csv(results_path)
results.columns = results.columns.str.strip()
best_mAP = results['metrics/mAP50(B)'].max()


# track
model_path = os.path.join(model_folder, 'weights', 'best.pt')
assert os.path.exists(model_path), 'Model file does not exist'
mota, motp, idf1, track_time, track_device = evaluate(
  model_path, 
  params['conf_threshold'], 
  params['iou_association_threshold'],
  params['imgsz'],
  params['tracker'],
  params['project_folder']
)

# Log on wandb
wandb.init(project="SharkTrack", name=params['name'], config=params, job_type="training")
wandb.log({'mAP': best_mAP, 'mota': mota, 'motp': motp, 'idf1': idf1, 'track_time': track_time, 'track_device': track_device})
wandb.finish()