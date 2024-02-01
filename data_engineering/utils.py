import os
import re
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_annotations(img_id, annotations_df):
    # allows for multiple bboxes per image
    # PASCAL VOC format
    return annotations_df.loc[annotations_df.Filename == img_id, 'Family Genus Species'.split()].values

def normalise_bbox(bbox, img_id, image_folder, img=None):
  if img is None:
    img = read_img(img_id, image_folder)
  img_h, img_w, _ = img.shape
  bbox = bbox / np.array([img_w, img_h, img_w, img_h])
  return bbox

def denormalise_bbox(bbox, img_id, image_folder, img=None):
  if img is None:
    img = read_img(img_id, image_folder)
  img_h, img_w, _ = img.shape
  bbox = bbox * np.array([img_w, img_h, img_w, img_h])
  return bbox


def draw_rect(img, bboxes, color=(255, 0, 0), bbox_relative=False):
    img = img.copy()
    if bbox_relative:
        bboxes = denormalise_bbox(bboxes, None, None, img)
    for bbox in bboxes:
        bbox = np.array(bbox).astype(int)
        pt1, pt2 = (bbox[0], bbox[1]), (bbox[2], bbox[3])
        img = cv2.rectangle(img, pt1, pt2, color, int(max(img.shape[:2]) / 200))
    return img

def read_img(img_id, image_folder):
    img_path = f'{image_folder}{img_id}'
    img = cv2.imread(str(img_path))
    return img

def read_bboxes(img_id, annotations_df, image_folder, bbox_relative=False):
    # allows for multiple bboxes per image
    bboxes = annotations_df.loc[annotations_df.Filename == img_id, 'xmin ymin xmax ymax'.split()].values
    if bbox_relative:
        bboxes = denormalise_bbox(bboxes, img_id, image_folder)
    return bboxes


def draw_bbox(img_id, image_folder, annotations_df, bbox_relative=False, img=None):
    if img is None:
        img = read_img(img_id, image_folder)
    bboxes = read_bboxes(img_id, annotations_df, image_folder, bbox_relative)
    if bbox_relative:
        # Coordinatew from 0 to 1
        bboxes = normalise_bbox(bboxes, None, None, img)
    img = draw_rect(img, bboxes)
    return img


def plot_img(img_id, image_folder, annotations_df=None, bbox=False):
    img = None
    if bbox:
        img = draw_bbox(img_id, image_folder, annotations_df)
    else:
        img = read_img(img_id, image_folder)

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb);
    
def plot_multiple_img(img_matrix_list, title_list, ncols, nrows=3, main_title=""):
    fig, myaxes = plt.subplots(figsize=(20, 15), nrows=nrows, ncols=ncols, squeeze=False)
    fig.suptitle(main_title, fontsize = 30)
    fig.subplots_adjust(wspace=0.3)
    fig.subplots_adjust(hspace=0.3)
    for i, (img, title) in enumerate(zip(img_matrix_list, title_list)):
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        myaxes[i // ncols][i % ncols].imshow(img_rgb)
        myaxes[i // ncols][i % ncols].set_title(title, fontsize=15)
        myaxes[i // ncols][i % ncols].grid(False)
        myaxes[i // ncols][i % ncols].set_xticks([])
        myaxes[i // ncols][i % ncols].set_yticks([])

    plt.show()