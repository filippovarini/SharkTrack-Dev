import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image_folder):
        self.image_folder = image_folder

        # Can find annotations in the image folder
        assert 'annotations.csv' in os.listdir(image_folder), 'annotations.csv not found in image_folder'
        self.annotations_df = pd.read_csv(os.path.join(image_folder, 'annotations.csv'))

    @staticmethod
    def normalise_bbox(bboxes, img):
        img_h, img_w, _ = img.shape
        norm_factors = np.array([img_w, img_h, img_w, img_h])
        normalised_bboxes = [bbox / norm_factors for bbox in bboxes]
        return normalised_bboxes

    @staticmethod
    def denormalise_bbox(bboxes, img):
        img_h, img_w, _ = img.shape
        denorm_factors = np.array([img_w, img_h, img_w, img_h])
        denormalised_bboxes = [bbox * denorm_factors for bbox in bboxes]
        return denormalised_bboxes
    
    @staticmethod
    def plot_multiple_img(img_matrix_list, title_list, ncols, nrows=3, main_title=""):
        fig, axes = plt.subplots(figsize=(10, 7.5), nrows=nrows, ncols=ncols, squeeze=False)
        fig.suptitle(main_title, fontsize=30)
        fig.subplots_adjust(wspace=0.3, hspace=0.3)
        for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax = axes[i // ncols, i % ncols]
            ax.imshow(img_rgb)
            ax.set_title(title, fontsize=15)
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()
    
    # Transformations
    @staticmethod
    def bgr2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    def read_annotations(self, img_id):
        return self.annotations_df.loc[self.annotations_df.Filename == img_id, 'Family Genus Species'.split()].values


    def draw_rect(self, img, bboxes, color=(255, 0, 0)):
        img = img.copy()
        for bbox in bboxes:
            bbox = np.array(bbox).astype(int)
            pt1, pt2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
            img = cv2.rectangle(img, pt1, pt2, color, int(max(img.shape[:2]) / 200))
        return img

    def read_img(self, img_id):
        img_path = os.path.join(self.image_folder, img_id)
        return cv2.imread(img_path)

    def read_bboxes(self, img_id):
        return self.annotations_df.loc[self.annotations_df.Filename == img_id, 'xmin ymin xmax ymax'.split()].values

    def draw_bbox(self, img_id, img=None):
        if img is None:
            img = self.read_img(img_id)
        bboxes = self.read_bboxes(img_id)
        if self.is_bbox_relative(img_id):
            bboxes = self.denormalise_bbox(bboxes, img)
        return self.draw_rect(img, bboxes)

    def is_bbox_relative(self, img_id):
        bboxes = self.read_bboxes(img_id)
        return np.all((bboxes >= 0) & (bboxes <= 1))

    def plot_img(self, img_id):
        img = self.read_img(img_id)
        img = self.draw_bbox(img_id, img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.show()
