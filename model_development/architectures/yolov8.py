from interfaces import Architecture
from ultralytics import YOLO
import utils
import pandas as pd
import numpy as np
import json
import os

class YoloV8(Architecture):
  def __init__(self, model_name, model_size, tracker, **kwargs):
    self.model_name = model_name
    self.tracker = tracker
    self.conf_treshold = kwargs['conf_treshold']
    self.iou_association_threshold = kwargs['iou_association_threshold']

    # If model is already trained, it would be in trained_models. Otherwise, train it from scratch.
    trained_models = None
    with open("../trained_models.json", "r") as file:
      trained_models = json.load(file)
    assert trained_models is not None
    if model_name in trained_models:
      path = trained_models[model_name]
      self.model = YOLO(self.path)

    else:
      assert model_size in ["n", "s", "m", "l", "x"]
      self.model = YOLO(f'yolov8{model_size}.pt')

    pass

  def __str__(self):
    return str(self.model)

  def train(self, arg1, arg2):
    """ Train the model using the dataloader provided, saves the model and updates
    the trained_models.json. """
    pass

  def evaluate(self):
    """
    1. Evaluate object detection model using the evaluation dataset
    2. Evaluate object detection model using the out-of-distribution evaluation dataset
    3. Evaluate tracker model using the evaluation dataset
    """
    videos = os.listdir(self.real_world_videos_folder)
    assert all([video.endswith('.mp4') for video in videos])
    annotations = os.listdir(self.real_world_annotations_folder)
    video_names = video_names = [vid[:-4] for vid in videos]
    assert len(videos) == len(annotations) and all([vid in annotations for vid in video_names])
    
    # 3. Evaluate tracker
    if self.tracker is not None:
      # macro average
      motas = []
      motps = []
      idf1s = []

      for video in video_names:
        print(f'Evaluating {video}')
        video_path = self.real_world_videos_folder + video
        annotations_path = self.real_world_annotations_folder + video[:-4] + '.csv'
        annotations = pd.read_csv(annotations_path)

        results = self.track(video_path)

        extracted_pred_results = self._extract_tracks(results)

        aligned_annotations = utils.align_annotations_with_predictions_dict_corrected(annotations, extracted_pred_results, self.real_world_video_length)
        mota, motp, idf1, frame_avg_motp = utils.evaluate_tracking(aligned_annotations, self.iou_association_threshold)
        print(f'{video} - MOTA: {round(mota, 2)}, MOTP: {round(motp, 2)}, IDF1: {round(idf1, 2)}')
        motas.append(mota)
        motps.append(motp)
        idf1s.append(idf1)

      macro_mota = round(np.mean(motas), 2)
      macro_motp = round(np.mean(motps), 2)
      macro_idf1 = round(np.mean(idf1s), 2)
      # TODO: construct average performance graph for each video! (or image of 6 combined)
    
    return macro_mota, macro_motp, macro_idf1

  def track(self, video_path):
    assert self.tracker is not None
    assert self.conf_treshold is not None and self.conf_treshold >= 0 and self.conf_treshold <= 1
    assert str(self.tracker) in ['botsort.yaml', 'bytetrack.yaml']

    results = self.model.track(
      source=video_path,
      persist=True,
      conf=self.conf_treshold,
      tracker=str(self.tracker)
    )
    return results
    
  def _extract_tracks(results):
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