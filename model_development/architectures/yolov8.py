from ultralytics.models.yolo.detect import DetectionTrainer
from data.dataloader_builder import DataLoaderBuilder
from data.yolo_dataset import YoloDataset
from ultralytics import YOLO, Trainer
from interfaces import Architecture
from types import SimpleNamespace
import pandas as pd
import numpy as np
import utils
import torch
import time
import json
import os


class CustomDetectionTrainer(DetectionTrainer):
    """
    This class is required to train the model with custom dataloaders
    """
    def __init__(self, *args, train_dataloader, val_dataloader, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_train_dataloader = train_dataloader
        self.custom_val_dataloader = val_dataloader

    # def build_dataset(self, img_path, mode="train", batch=None):
    #     # Simply return None as we are not building a dataset here
    #     return None

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        # Use the custom dataloaders instead of constructing new ones
        if mode == "train":
            return self.custom_train_dataloader
        elif mode == "val":
            return self.custom_val_dataloader
        else:
            raise ValueError("Mode must be 'train' or 'val', we don't allow testing as we do out-of-sample testing")


class YoloV8(Architecture):
  def __init__(self, hyperparameters, tracker):
    super().__init__()
    self.hyperparameters = hyperparameters
    self.tracker = tracker
    
    if self.hyperparameters["pretrained"]:
      self.model = YOLO(self.hyperparameters['model_path'])
    else:
      # TODO: set model path
      assert self.hyperparameters["model_size"] in ["n", "s", "m", "l", "x"]
      self.model = YOLO(f'yolov8{self.hyperparameters["model_size"]}.pt')

    print(f"Initialised Model {self.hyperparameters['model_name']} ")

  def __str__(self):
    return str(self.model)

  def train(self, dataset: YoloDataset):
    """ 
    Train the model using the dataset provided

    :param dataset: Dataset object containing the training data, extending torch.utils.data.Dataset
    """
    builder = DataLoaderBuilder(dataset, self.hyperparameters['batch_size'])
    train_loader, val_loader, test_loader = builder.build()
    model_folder = os.path.dirname(self.hyperparameters['model_path'])

    trainer_params = {
      "epochs": self.hyperparameters['epochs'],
      "model": self.model,
      "save_dir": model_folder,
      "wdir": model_folder,
      "batch_size": self.hyperparameters['batch_size'],
      "device": utils.get_torch_device(),
    }

    trainer = CustomDetectionTrainer(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        **trainer_params
      )

    trainer.args.imgsz = self.hyperparameters['img_size']
    trainer.args.batch_size = self.hyperparameters['batch_size'],
    trainer.args.epochs = self.hyperparameters['epochs'],
    trainer.args.verbose = True,
    trainer.args.patience = self.hyperparameters['patience'],
    trainer.args.lr0 = self.hyperparameters['learning_rate'],
    trainer.args.lr1 = self.hyperparameters['learning_rate'],
    
    results = trainer.train()
    self.model = trainer.best

  def evaluate(self):
    """
    1. Evaluate object detection model using the evaluation dataset
    2. Evaluate object detection model using the out-of-distribution evaluation dataset
    3. Evaluate tracker model using the evaluation dataset
    """
    bruvs_video_folder = self.bruvs_videos_folder if not self.hyperparameters['greyscale'] else self.greyscale_bruvs_videos_folder
    videos = os.listdir(bruvs_video_folder)
    assert all([video.endswith('.mp4') for video in videos])
    annotations = os.listdir(self.bruvs_annotations_folder)
    video_names = video_names = [vid[:-4] for vid in videos]
    assert len(videos) == len(annotations) and all([f'{vid}.csv' in annotations for vid in video_names])
    
    # 3. Evaluate tracker
    track_start_time = time.time()
    if self.tracker is not None:
      # macro average
      motas = []
      motps = []
      idf1s = []

      for video in video_names:
        print(f'Evaluating {video}')
        video_path = bruvs_video_folder + video + '.mp4'
        annotations_path = self.bruvs_annotations_folder + video + '.csv'
        annotations = pd.read_csv(annotations_path)

        results = self.track(video_path)

        # Extract and store annotations for investigation
        extracted_pred_results = self._extract_tracks(results)
        aligned_annotations = utils.align_annotations_with_predictions_dict_corrected(annotations, extracted_pred_results, self.bruvs_video_length)
        aligned_annotations_df = pd.DataFrame(aligned_annotations)
        aligned_annotations_path = self.hyperparameters['annotation_path']
        aligned_annotations_df.to_csv(aligned_annotations_path, index=False)

        mota, motp, idf1, frame_avg_motp = utils.evaluate_tracking(aligned_annotations, self.hyperparameters['iou_association_threshold'])
        print(f'{video} - MOTA: {round(mota, 2)}, MOTP: {round(motp, 2)}, IDF1: {round(idf1, 2)}')
        motas.append(mota)
        motps.append(motp)
        idf1s.append(idf1)

      macro_mota = round(np.mean(motas), 2)
      macro_motp = round(np.mean(motps), 2)
      macro_idf1 = round(np.mean(idf1s), 2)
      # TODO: construct average performance graph for each video! (or image of 6 combined)

    track_end_time = time.time()
    track_time = round((track_end_time - track_start_time) / 60, 2)

    return macro_mota, macro_motp, macro_idf1, track_time, utils.get_torch_device()

  def track(self, video_path):
    assert self.tracker is not None
    assert str(self.tracker) in ['botsort.yaml', 'bytetrack.yaml']

    results = self.model.track(
      source=video_path,
      persist=True,
      conf=self.hyperparameters['conf_threshold'],
      iou=self.hyperparameters['iou_association_threshold'],
      imgsz=self.hyperparameters['img_size'],
      tracker=str(self.tracker)
    )
    return results
    
  def _extract_tracks(self, results):
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