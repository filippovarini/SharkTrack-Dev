import albumentations as A
from data.image_processor import ImageProcessor
import numpy as np
import cv2

def bbox_only_rotate(img, bboxes):
    """ 
    Rotates only the bounding box in the image
    """
    bbox_params = {'format': 'pascal_voc', 'label_fields': ['labels']}
    aug = A.Compose([A.Rotate(p=1, limit=90)], bbox_params=bbox_params)
    anno = aug(image=img, bboxes=bboxes, labels=np.ones(len(bboxes)))
    new_bboxes = np.array(anno['bboxes'])
    rotated_image = anno['image']
    
    # Extract the bboxed images from new_image and paste them in the original img
    # Create a copy of the original image to paste the rotated regions
    result_image = img.copy()

    # Iterate over the original and new bounding boxes
    for original_bbox, new_bbox in zip(bboxes, new_bboxes):
        # Extract coordinates
        x_min, y_min, x_max, y_max = [int(coord) for coord in original_bbox]
        new_x_min, new_y_min, new_x_max, new_y_max = [int(coord) for coord in new_bbox]

        # Extract the rotated region from the rotated image
        rotated_region = rotated_image[new_y_min:new_y_max, new_x_min:new_x_max]

        # Resize the rotated region to fit the original bbox size
        # original_width, original_height = x_max - x_min, y_max - y_min
        # resized_rotated_region = cv2.resize(rotated_region, (original_width, original_height))

        # Paste the resized rotated region onto the original image
        # result_image[y_min:y_max, x_min:x_max] = resized_rotated_region
        result_image[new_y_min:new_y_max, new_x_min:new_x_max] = rotated_region
    
    return result_image, new_bboxes

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