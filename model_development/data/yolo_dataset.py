# Subclass of data.Dataset, just to wrap the __getitem__ method to return YOLO format!
from data.dataset import Dataset
from data.image_processor import ImageProcessor

class YoloDataset(Dataset):
  def __init__(self, root_dir, subfolder_sampling_ratios, augmentations=[]):
    super().__init__(root_dir, subfolder_sampling_ratios, augmentations)

  def _to_yolo(self, bboxes):
    """
    input: bbox: [[xmin, ymin, xmax, ymax], ...]
    output: [[class, x_center, y_center, width, height], ...]
    """
    yolo_bboxes = []
    for bbox in bboxes:
      xmin, ymin, xmax, ymax = bbox
      width, height = xmax - xmin, ymax - ymin
      x_center, y_center = xmin + width / 2, ymin + height / 2
      yolo_bboxes.append([0, x_center, y_center, width, height])
    return yolo_bboxes
  
  def __getitem__(self, idx):
    # Use the parent class's __getitem__ method to get the image and annotations
    # and update the annotations to be in YOLO format
    image, bboxes = super().__getitem__(idx)
    normalised_bboxes = ImageProcessor.normalise_bbox(bboxes, image)
    yolo_bboxes = self._to_yolo(normalised_bboxes)
    return image, yolo_bboxes

