from data.advanced_augmentations import apply_custom_cutout
from data.image_processor import ImageProcessor
from torch.utils.data import Dataset
import albumentations as A
import pandas as pd
import numpy as np
import random
import torch
import cv2
import os

ALLOWED_AUGMENTATIONS = ["Equalise", "Rotate", "Crop", "Cutout"]

class CustomDataset(Dataset):
    experimentation_dataset_path = '/vol/biomedic3/bglocker/ugproj2324/fv220/datasets/experimentation_datasets'

    def __init__(self, dataset_name, root_dir, subfolder_sampling_ratios, augmentations=[], transforms=[], **kwargs):
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
        self.subfolders = subfolder_sampling_ratios.keys()
        self.dataset_name = dataset_name

        assert all([os.path.isdir(os.path.join(root_dir, folder)) for folder in self.subfolders]), \
            'root_dir should contain only directories'
        assert len(augmentations) == 0 or [aug in ALLOWED_AUGMENTATIONS for aug in augmentations], \
            f'Augmentations should be one of {ALLOWED_AUGMENTATIONS}'
        
        self.root_dir = root_dir
        self.augmentations = augmentations
        self.subfolder_sampling_ratios = subfolder_sampling_ratios
        self.img_size = kwargs['img_size'] # size of image the model takes
        self.dataset_size = self._inspect_dataset_size()
        self.image_paths  = self._load_paths()
        self.transforms = transforms
    
    def _inspect_dataset_size(self):
        # Print number of images in each subfolder
        dataset_size = {}
        for folder in self.subfolders:
            folder_path = os.path.join(self.root_dir, folder)
            original_size = len([f for f in os.listdir(folder_path) if self._file_is_image(f)])
            dataset_size[folder] = original_size
            sampling_ratio = self.subfolder_sampling_ratios[folder]
            dataset_size[folder] = int(original_size * sampling_ratio)
        return dataset_size
    
    def get_img_processor(self, idx):
        img_path = self.image_paths[idx]
        image_folder = os.path.dirname(img_path)
        image_id = os.path.basename(img_path)
        return ImageProcessor(image_folder), image_id
    
    def _is_bbox_relative(self, idx, bboxes=None):
        if bboxes:
            if not isinstance(bboxes, np.ndarray):
                bboxes = np.array(bboxes)
            return np.all((bboxes >= 0) & (bboxes <= 1))
        else:
            image_processor, image_id = self.get_img_processor(idx)
            return image_processor.is_bbox_relative(image_id)
    
    def plot_single_image(self, idx):
        image_processor, image_id = self.get_img_processor(idx)
        print(image_id)
        image_processor.plot_img(image_id)
    
    def get_info(self, model_folder=None):
        sample = random.sample(range(len(self.image_paths)), 9)
        boxed_images = []
        for i in sample:
            annotation = self[i]
            img = annotation['image']
            bboxes = annotation['bboxes']
            # print(f'returned by self[] {i}, {bboxes}')
            # if self._is_bbox_relative(i, bboxes):
            #     print(f'denormalising {i}, {bboxes}')
            #     bboxes = ImageProcessor.denormalise_bbox(bboxes, img)
            # print(f'after denormalisation {i}, {bboxes}')
            boxed_images.append(ImageProcessor.draw_rect(img, bboxes))

        save_fig = model_folder is not None

        fig = ImageProcessor.plot_multiple_img(
            boxed_images,
            [str(s) for s in sample],
            ncols=3,
            nrows=3,
            main_title="Dataset Sample",
            return_fig=save_fig
        )

        if save_fig:
            # If model_folder is not a folder, take the parent folder
            if not os.path.isdir(model_folder):
                print(f'{model_folder} is not a folder. Taking parent folder instead.')
                model_folder = os.path.dirname(model_folder)
            fig_path = os.path.join(model_folder, 'dataset_sample.png')
            print(f'Saving dataset sample to {fig_path}')
            fig.savefig(fig_path)
    
    def _file_is_image(self, file):
        return file.endswith('.jpg') or file.endswith('.png') or file.endswith('.jpeg')
    
    def _sample_images(self, image_paths, folder):
        """
        User can specify the percentage of images to sample from each subfolder.
        In this case, sample the images.
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
            subfolder_path = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if self._file_is_image(file)]

            subfolder_images = self._sample_images(subfolder_path, folder)
            assert len(subfolder_images) == self.dataset_size[folder], \
                f'Number of images should be equal to the dataset size. Got {len(subfolder_images)} images and dataset size {self.dataset_size[folder]}.' 
            
            image_paths += subfolder_images

        return image_paths
    
    def _augment(self, img, bboxes):
        """
        Composes all the augmentations specified. Making sure that there is 
        equal probability of each augmentation being applied and none of them
        """
        assert len(self.augmentations) > 0, 'Augmentations should not be empty'
        standard_augmentations = []
        p = 1 / len(self.augmentations)
        bbox_params = {'format': 'pascal_voc', 'label_fields': ['labels']} 

        if 'Equalise' in self.augmentations:
            standard_augmentations.append(A.Equalize(p=p))
        if 'Rotate' in self.augmentations:
            standard_augmentations.append(A.Rotate(limit=90, p=p))
        if 'Crop' in self.augmentations:
            h = img.shape[0]
            w = img.shape[1]
            standard_augmentations.append(A.RandomCrop(p=p, height=int(h*0.8), width=int(w * 0.8)))

        albumentation_pipeline = A.Compose(standard_augmentations, bbox_params=bbox_params)

        labels = np.ones(len(bboxes)) # Works for single "shark class"
        # TODO: extend for species-level classifier classes
        aug = albumentation_pipeline(image=img, bboxes=bboxes, labels=labels)
        aug_img, aug_bboxes = aug['image'], aug['bboxes']

        # BBoxes are list of tuples. Turn them in 2d numpy array
        aug_bboxes = np.array(aug_bboxes)
        
        # Some Augmentations are not available in Albumentations, so we have 
        # to define them ourselves. Usually, they are applied on the bounding
        # box, so we can run them only if the image has bounding boxes.
        if 'Cutout' in self.augmentations and np.random.rand() < p and len(aug_bboxes) > 0:
            assert all([np.any(np.array(bbox) > 1) for bbox in aug_bboxes]), 'Bbox coordinates must not be normalised'
            print('cutout')
            aug_img, aug_bboxes = apply_custom_cutout(aug_img, bboxes=aug_bboxes)

        assert type(aug_bboxes) == np.ndarray, 'Bboxes should be a numpy array'
        
        return aug_img, aug_bboxes
    
    def _apply_transforms(self, img):
        for transform in self.transforms:
            img = transform(img)
        return img
        
    def __len__(self):
        dataset_length = sum(self.dataset_size.values())
        assert dataset_length == len(self.image_paths), \
            f'Dataset length should be equal to the number of images. Got {dataset_length} and {len(self.image_paths)} images.'
        return dataset_length

    def __getitem__(self, idx):
        """
        Returns image and bboxes in the following format:
        - pascal-voc
        - non-normalised
        """
        image_path = self.image_paths[idx]

        image_folder = os.path.dirname(image_path)
        image_id = os.path.basename(image_path)
        image_processor = ImageProcessor(image_folder)

        image = image_processor.read_img(image_id)
        image = self._apply_transforms(image)
        
        bboxes = image_processor.read_bboxes(image_id)
        # print(f'{idx}, {bboxes}')
        if image_processor.is_bbox_relative(image_id):
            bboxes = ImageProcessor.denormalise_bbox(bboxes, image)

        print(idx)
        if len(self.augmentations) > 0:
            aug_img, aug_bboxes = self._augment(image, bboxes)
            image, bboxes = aug_img, aug_bboxes
            assert len(bboxes) == 0 or np.array(bboxes).shape[1] == 4, 'Bboxes should be in pascal-voc format'

        return {"image": image, "bboxes": bboxes}
