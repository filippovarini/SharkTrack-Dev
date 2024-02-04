import albumentations as A
from data.image_processor import ImageProcessor
import numpy as np
import cv2

def apply_custom_cutout(image, bboxes, fill_color=(130, 160, 160)):
    """
    Applies cutout on the bounding box, to simulate occlusion.

    :param image: A numpy array representing the RGB image.
    :param bboxes: A list of bounding boxes in relative coordinates (xmin, ymin, xmax, ymax).
    :param max_height: Maximum height of the cutout.
    :param max_width: Maximum width of the cutout.
    :param fill_color: RGB color for the cutout area.
    :return: Augmented image.
    """
    h, w = image.shape[:2]
    bbox_index = np.random.randint(0, len(bboxes)) # pick a random bbox to apply cutout to
    bbox = bboxes[bbox_index]

    # Convert relative bbox to absolute coordinates
    xmin, ymin, xmax, ymax = [int(bbox[i]) for i in range(4)]

    # Random center point within the bbox
    center_x = np.random.randint(xmin, xmax)
    center_y = np.random.randint(ymin, ymax)

    # Max width and max heigth is equal to max(1.5x bbox width, image_width) and max(1.5x bbox height, image_height)
    max_height = min(int(1.5 * (ymax - ymin)), h)
    max_width = min(int(max_height * 0.5), w)

    min_height = min(int((ymax - ymin)), h)
    min_height = min(min_height, max_height - 1)
    min_width = min(int(0.3 * (xmax - xmin)), max_width - 1)


    # Random height and width of the cutout
    cutout_width = np.random.randint(min_width, max_width)
    cutout_height = np.random.randint(min_height, max_height)

    # Calculating the top left corner of the cutout
    x1 = max(center_x - cutout_width // 2, 0)
    y1 = max(center_y - cutout_height // 2, 0)

    # Ensure cutout does not exceed bbox boundaries
    x2 = min(x1 + cutout_width, w)
    y2 = min(y1 + cutout_height, h)

    # Apply cutout
    image[y1:y2, x1:x2] = fill_color

    return image, bboxes