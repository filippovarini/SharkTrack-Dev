import pandas as pd
import cv2
import os

def clip_bbox_xyxy(annotation_path, image_folder):
    print(f'Clipping bbox xyxy of {annotation_path}...')
    annotations_df = pd.read_csv(annotation_path)

    # Assume all images have same dimensions
    sample_img = next(f for f in os.listdir(image_folder) if f.endswith('.jpg'))
    img = cv2.imread(os.path.join(image_folder, sample_img))
    height, width, _ = img.shape

    annotations_df['xmax'] = annotations_df['xmax'].clip(0, width)
    annotations_df['ymax'] = annotations_df['ymax'].clip(0, height)
    annotations_df['xmin'] = annotations_df['xmin'].clip(0, width)
    annotations_df['ymin'] = annotations_df['ymin'].clip(0, height)

    annotations_df.to_csv(annotation_path, index=False)

def clip_dataset_bbox_xyxy(dataset_path):
    for source in os.listdir(dataset_path):
        annotation_path = os.path.join(dataset_path, source, 'annotations.csv')
        image_folder = os.path.join(dataset_path, source)
        clip_bbox_xyxy(annotation_path, image_folder)
