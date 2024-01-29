from image_processor import ImageProcessor
from torch.utils.data import Dataset
from advanced_augmentations import apply_custom_cutout, apply_custom_rotate
import albumentations as A
import pandas as pd
import numpy as np
import torch
import cv2
import os

REQUIRED_ANNOTATION_COLS = ["Filename", "Family", "Genus", "Species", "ymin", "xmin", "xmax", "ymax", "Augmentation", "Source"]
ALLOWED_AUGMENTATIONS = ["Equalise", "Rotate", "Crop", "Bbox-rotate", "Cutout"]
MAX_CROP = 150


class CustomDataset(Dataset):
    def __init__(self, root_dir, subfolder_sampling_ratios, augmentations=None):
        """
        Note:
        - Augmentations don't change the number of images in the dataset, as they are applied on the fly.
        - Every image can have only one augmentation
        - If augmentations are specified, they are applied (with a probability) to every image in the dataset.

        root_dir: Path to the 'datasets/' folder.
        subfolder_sampling_ratios: Dict[str, float] where the key is the name of the subfolder and the value is the percentage
          of images to sample.
        augmentations: List of albumentations augmentations to apply.
        """
        self.subfolders = os.listdir(root_dir)

        # Assert the root_dir contains only directories and no files
        assert all([os.path.isdir(os.path.join(root_dir, folder)) for folder in self.subfolders]), \
            'root_dir should contain only directories'
        assert [aug in ALLOWED_AUGMENTATIONS for aug in augmentations], \
            f'Augmentations should be one of {ALLOWED_AUGMENTATIONS}'
        
        self.root_dir = root_dir
        self.augmentations = augmentations
        self.subfolder_sampling_ratios = subfolder_sampling_ratios
        self.dataset_size = self._inspect_dataset_size()
        self.image_paths  = self._load_paths()
    
    def _inspect_dataset_size(self):
        # Print number of images in each subfolder
        dataset_size = {}
        for folder in self.subfolders:
            folder_path = os.path.join(self.root_dir, folder)
            original_size = len(os.listdir(folder_path))
            dataset_size[folder] = original_size
            sampling_ratio = self.subfolder_sampling_ratios[folder]
            dataset_size[folder] = int(original_size * sampling_ratio)
            print(f'{folder}: original size: {original_size}, samples: {dataset_size[folder]} images')
        return dataset_size
    
    def get_info(self):
        # TODO
        pass

    def show_image(self, source, image_name):
        # TODO
        pass
    
    def _file_is_image(self, file):
        return file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')
    
    
    def _sample_images(self, image_paths, annotations_df, folder):
        """
        User can specify the percentage of images to sample from each subfolder.
        In this case, sample the images and adjust annotations accordingly, by 
        only having annotations for the sampled images.
        """
        filenames = [os.path.basename(image_path) for image_path in image_paths]

        num_files = len(filenames)
        sampling_ratio = self.subfolder_sampling_ratios[folder]
        num_files_to_sample = int(num_files * sampling_ratio)

        sampled_filenames = np.random.choice(filenames, num_files_to_sample, replace=False)
        sampled_image_paths = [os.path.join(self.root_dir, folder, filename) for filename in sampled_filenames]
        return sampled_image_paths

    def _load_paths(self):
        image_paths = []
        
        for folder in self.subfolders:
            folder_path = os.path.join(self.root_dir, folder)
            image_paths += [os.path.join(folder_path, file) for file in os.listdir(folder_path) if self._file_is_image(file)]

            # Filter image_paths based on the sampling ratio
            if folder in self.subfolder_sampling_ratios:
                image_paths = self._sample_images(image_paths, folder)

            assert len(image_paths) == self.dataset_size[folder], \
                f'Number of images should be equal to the dataset size. Got {len(image_paths)} images and dataset size {self.dataset_size[folder]}.' 

        return image_paths
    
    def _augment(self, img, bboxes):
        """
        Composes all the augmentations specified. Making sure that there is 
        equal probability of each augmentation being applied and none of them
        """
        standard_augmentations = []
        p = 1 / len(self.augmentations)
        bbox_params = {'format': 'pascal_voc', 'label_fields': ['labels']} 

        if 'Equalise' in self.augmentations:
            standard_augmentations.append(A.Equalize(p=p))
        if 'Rotate' in self.augmentations:
            standard_augmentations.append(A.Rotate(limit=90, p=p))
        if 'Crop' in self.augmentations:
            standard_augmentations.append(A.RandomCrop(p=p, height=MAX_CROP, width=MAX_CROP))

        albumentation_pipeline = A.compose(standard_augmentations, bbox_params=bbox_params)

        labels = np.ones(len(bboxes)) # Works for single "shark class"
        # TODO: extend for species-level classifier classes
        aug = albumentation_pipeline(image=img, bboxes=bboxes, labels=labels)
        aug_img, aug_bboxes = aug['image'], aug['bboxes']
        
        # Some Augmentations are not available in Albumentations, so we have to define them ourselves
        if 'Cutout' in self.augmentations and np.random.rand() < p:
            assert any([coord > 1 for bbox in aug_bboxes for coord in bbox]), 'Bbox coordinates must not be relative'
            aug_img, aug_bboxes = apply_custom_cutout(aug_img, bboxes=aug_bboxes)
        if 'Bbox-rotate' in self.augmentations and np.random.rand() < p:
            aug_img, aug_bboxes = apply_custom_rotate(aug_img, bboxes=aug_bboxes)
        
        return aug_img, aug_bboxes
        
    def __len__(self):
        dataset_length = sum(self.dataset_size.values())
        assert dataset_length == len(self.image_paths), \
            f'Dataset length should be equal to the number of images. Got {dataset_length} and {len(self.image_paths)} images.'
        return dataset_length

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        image_folder = os.path.dirname(image_path)
        image_id = os.path.basename(image_path)
        image_processor = ImageProcessor(image_folder)

        image = image_processor.read_img(image_id)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bboxes = image_processor.read_bboxes(image_id)

        if self.augmentations:
            self._augment(image, bboxes)
            aug_img, aug_bboxes = self.augmentations(image=image)
            image, bboxes = aug_img, aug_bboxes

        return image, bboxes
