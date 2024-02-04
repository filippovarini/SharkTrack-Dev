# Subclass of data.Dataset, just to wrap the __getitem__ method to return YOLO format!
from data.dataset import CustomDataset
from data.image_processor import ImageProcessor
from data.dataloader_builder import DataLoaderBuilder
import random
import shutil
import yaml
import cv2
import os

class YoloDataset(CustomDataset):
  def __init__(self, dataset_name, root_dir, subfolder_sampling_ratios, augmentations=[], transforms=[], **kwargs):
    super().__init__(dataset_name, root_dir, subfolder_sampling_ratios, augmentations,  transforms=[], **kwargs)
    self.classes = {0: 'shark'} # single class object detection

  def construct_classes(self):
    # for multiclass, return a list of classes and names it maps to
    pass

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
    istance = super().__getitem__(idx)
    image, bboxes = istance['image'], istance['boxes']
    normalised_bboxes = ImageProcessor.normalise_bbox(bboxes, image)
    yolo_bboxes = self._to_yolo(normalised_bboxes)
    return {"image": image, "boxes": yolo_bboxes}
  
  def build(self):
    """
    Yolo wants the data to be stored in a specific folder structure and to pass
    the path. We could do wiht a dataloader but this approach is leaner.
    # dataset_path
        # ├── train
        # │   ├── images
        # │   └── labels
        # ├── val
        # │   ├── images
        # │   └── labels
        # └── test
        #     ├── images
    """
    assert self.dataset_name not in os.listdir(self.experimentation_dataset_path), f"Dataset {self.dataset_name} already exists."

    dataset_path = os.path.join(self.experimentation_dataset_path, self.dataset_name)
    # TODO: static field now
    try:
      train_ratio, val_ratio, test_ratio = DataLoaderBuilder.get_split_ratios()
      print(f'Building dataset {self.dataset_name} in {dataset_path} by copying {len(self)} images...')

      # Create dataset folder
      os.mkdir(dataset_path)
      # Create subfolders
        
      # Calculate indices for train, val and test
      indices = list(range(len(self)))
      random.shuffle(indices)
      train_size = int(train_ratio * len(self))
      val_size = int(val_ratio * len(self))
      test_size = len(self) - train_size - val_size
      split = {
        'train': indices[:train_size],
        'val': indices[train_size:train_size+val_size],
        'test': indices[train_size+val_size:]
      }

      subfolders = ['train', 'val', 'test']
      for subfolder in subfolders:
        print(f'Creating subfolder "{subfolder}" of size {len(split[subfolder])} ...')
        os.mkdir(os.path.join(dataset_path, subfolder))
        images_path = os.path.join(dataset_path, subfolder, 'images')
        labels_path = os.path.join(dataset_path, subfolder, 'labels')
        os.mkdir(images_path)
        os.mkdir(labels_path)

        # Copy images and create labels
        # Note that when we get an image, we perform augmentations and get back the 
        # augmented image and relative bboxes. Therefore, we need to copy the augmented image
        # not, the original.
        # However, if augmentation is [], then we don't perform any augmentation,
        # so we can directly use shutil.copyfile
        for i in split[subfolder]:
          annotations = self[i]
          image, bboxes = annotations['image'], annotations['bboxes']
          image_path = self.image_paths[i]
          image_id = os.path.basename(image_path)
          new_image_path = os.path.join(images_path, image_id)
          if len(self.augmentations) > 0:
            # Write image represented by numpy tensor to new_image_path
            cv2.imwrite(new_image_path, image)
          else:
            # simply copy original image
            shutil.copyfile(image_path, new_image_path)

          # Create label
          label_id = os.path.splitext(image_id)[0] + '.txt'
          label_path = os.path.join(labels_path, label_id)
          with open(label_path, 'w') as f:
            for bbox in bboxes:
              f.write(' '.join([str(round(b, 4)) for b in bbox]) + '\n')

      # Add data_config.yaml file with Yolo Format
      config = {
        'path': dataset_path,
        'train': './train',
        'val': './val',
        'test': './test',
        'names': self.classes,
        'data_split': self.subfolder_sampling_ratios,
        'augmentations': self.augmentations
      }
      with open(os.path.join(dataset_path, 'data_config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    except Exception as e:
      print(f'Error while building dataset {self.dataset_name}: {e}')
      shutil.rmtree(dataset_path)
      raise e

    return dataset_path

        


    



