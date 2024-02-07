from abc import ABC, abstractmethod
import os

class Architecture(ABC):
    """
    Abstract class for architecture, which takes configuration parameters, dataset
    and tracker and trains the model, returning a new model.
    If the model is already present, it loads the model and returns it.
    """
    def __init__(self):
        super().__init__()
        self.bruvs_val_folder = '/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/validation/val2/'
        assert 'frames' in os.listdir(self.bruvs_val_folder), 'The folder should contain a folder called frames'
        assert 'annotations' in os.listdir(self.bruvs_val_folder), 'The folder should contain a folder called annotations'
        self.bruvs_annotation_fps = 1
        self.bruvs_video_length = 20

    @abstractmethod
    def __str__(self):
        """ Prints the model architecture"""
        pass

    @abstractmethod
    def train(self, arg1, arg2):
        """ Trains the model, evaluating the loss on the validation set.
        Saves the best model according to the validation loss."""
        pass
    
    @abstractmethod
    def evaluate(self, tracker):
        """ Evaluates two models using the evaluation dataset:
        1. Object detection model, returning mAP@50 and F1
            this should be against val dataset and out-of-distribution val dataset
        2. If tracker=True, returns the MOTA, MOTP and IDF1 of the tracker"""
        pass
    
    @abstractmethod
    def track_frames(self, tracker_model):
        """ Runs the tracker model on top of the object detection model."""
        pass

    
class Tracker(ABC):
    @abstractmethod
    def update(self, detections):
        """ Update the tracks and returns the updated tracks."""
        pass
    
    @abstractmethod
    def __str__(self):
        pass