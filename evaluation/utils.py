import matplotlib.pyplot as plt
import pandas as pd
import torch
import os

def extract_frame_number(frame_name):
  assert frame_name.endswith('.jpg')
  return int(frame_name.replace('.jpg', '').split('_')[-1].replace('frame', ''))


def target2pred_align(annotations, track_predictions, sequence_path, tracker):
    """
    Predictions are definitive, target to align 
    """
    tot_annotation_frames = len(track_predictions[0])
    gt_bbox_xyxys = [[]] * tot_annotation_frames
    gt_track_ids = [[]] * tot_annotation_frames

    frames = [f for f in os.listdir(sequence_path) if f.endswith('.jpg')]
    frames.sort(key=extract_frame_number)

    for _, frame_annotations in annotations.groupby('Filename'):
        frame_name = frame_annotations["Filename"].values[0]
        i = frames.index(frame_name)

        gt_bbox_xyxys[i] = frame_annotations[["xmin", "ymin", "xmax", "ymax"]].values.tolist()
        gt_track_ids[i] = frame_annotations["track_id"].values.tolist()
    
    results = {
        "gt_bbox_xyxys": gt_bbox_xyxys,
        "gt_track_ids": gt_track_ids,
        "pred_bbox_xyxys": track_predictions[0],
        "pred_confidences": track_predictions[1],
        "pred_track_ids": track_predictions[2],
        "frame_id": [i for i in range(tot_annotation_frames)]
    }

    assert len(results["gt_bbox_xyxys"]) == tot_annotation_frames
    assert len(results["gt_track_ids"]) == tot_annotation_frames
    assert len(results["pred_bbox_xyxys"]) == tot_annotation_frames
    assert len(results["pred_confidences"]) == tot_annotation_frames
    if tracker is not None:
        # when we don't have tracker we run detection only, so we don't have track_ids
        assert len(results["pred_track_ids"]) == tot_annotation_frames

    return results    



def align_annotations_with_predictions_dict_corrected(annotations, track_predictions, video_length):
    """
    Correctly aligns ground truth annotations with predicted data from an object tracking model.
    Each row in the annotations represents a detection, not necessarily a frame.

    :param annotations: DataFrame with ground truth annotations. It must have the following columns:
                        - frame_id: Frame number
                        - track_id: Unique ID for the track
                        - xmin, ymin, xmax, ymax: Bounding box coordinates
    :param track_predictions: List of predictions in the format [[bbox_xyxy], [confidences], [track_ids]] for each frame
    :param video_length: Length of the video in seconds.
    :return: List of aligned data in dictionary format.
    """
    # Ground truth frame rate is given as 10 FPS
    gt_frame_rate = 10
    tot_annotation_frames = gt_frame_rate * video_length

    # Calculate the predicted frame rate
    tot_pred_frames = len(track_predictions[0])
    pred_frame_rate = tot_pred_frames / video_length

    assert tot_annotation_frames <= tot_pred_frames # orig video > 10fps

    # Initialize the output list
    results = {
        "gt_bbox_xyxys": [],
        "gt_track_ids": [],
        "pred_bbox_xyxys": [],
        "pred_confidences": [],
        "pred_track_ids": []
    }


    for frame_num in range(tot_annotation_frames):
        ### GET PRED FRAME TRACKS
        # Calculate the corresponding frame in the predictions
        pred_frame_index = int(round(frame_num * pred_frame_rate / gt_frame_rate))
        assert pred_frame_index < tot_pred_frames

        # Extract predicted data for the corresponding frame
        pred_bbox_xyxys = track_predictions[0][pred_frame_index]
        pred_confidences = track_predictions[1][pred_frame_index]
        pred_track_ids = track_predictions[2][pred_frame_index]

        ### GET GT FRAME TRACKS
        # Filter annotations dataframe that has frame_id = frame_num
        frame_annotations = annotations[annotations["frame_id"] == frame_num]

        # Extract ground truth data for the corresponding frame
        gt_track_ids = frame_annotations["track_id"].values.tolist()
        gt_bbox_xyxys = frame_annotations[["xmin", "ymin", "xmax", "ymax"]].values.tolist()

        results["gt_bbox_xyxys"].append(gt_bbox_xyxys)
        results["gt_track_ids"].append(gt_track_ids)
        results["pred_bbox_xyxys"].append(pred_bbox_xyxys)
        results["pred_confidences"].append(pred_confidences)
        results["pred_track_ids"].append(pred_track_ids)

    assert len(results["gt_bbox_xyxys"]) == tot_annotation_frames
    assert len(results["gt_track_ids"]) == tot_annotation_frames
    assert len(results["pred_bbox_xyxys"]) == tot_annotation_frames
    assert len(results["pred_confidences"]) == tot_annotation_frames
    assert len(results["pred_track_ids"]) == tot_annotation_frames

    return results

def calculate_iou(box_a, box_b):
    # Calculate the (x, y)-coordinates of the intersection rectangle
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    boxBArea = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    # Compute the intersection over union
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_tracking(results, S_TRESH):
    """
    Calculates MOTA, MOTP and IDF1 metrics for a given set of results.
    """
    total_misses = 0
    total_false_positives = 0
    total_mismatches = 0
    total_ground_truth = 0
    frame_avg_motp = [] # Frame sum of IoU / number of GT bboxes in frame
    tot_iou = 0
    idtp = 0
    # This is not counted in MOTA because MOTA focuses on detection
    idf1_extra_fn = 0 # When ID switch, the lost, correct old track is counted as FN
    idf1_extra_fp = 0 # When ID switch, the new tra kis a FP

    id_mapping = {}  # Maps predicted IDs to ground truth IDs
    id_switched = {} # Maps ground truth IDs to predicted IDs before id switch

    tot_frames = len(results["gt_bbox_xyxys"])

    for frame_idx in range(tot_frames):
        gt_bboxes = results["gt_bbox_xyxys"][frame_idx]
        gt_ids = results["gt_track_ids"][frame_idx]
        pred_bboxes = results["pred_bbox_xyxys"][frame_idx]
        pred_ids = results["pred_track_ids"][frame_idx]
        total_ground_truth += len(gt_bboxes)
        frame_tot_iou = 0

        matches = {}  # Maps ground truth IDs to predicted IDs for this frame

        # Find matches and calculate mismatches
        for i, gt_box in enumerate(gt_bboxes):
            gt_id = gt_ids[i]
            best_iou = S_TRESH
            best_pred_idx = -1

            for j, pred_box in enumerate(pred_bboxes):
                iou = calculate_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = j

            if best_pred_idx != -1:
                pred_id = pred_ids[best_pred_idx]
                matches[gt_id] = pred_id
                frame_tot_iou += best_iou

                if gt_id in id_switched:
                    if id_switched[gt_id] != pred_id:
                        idf1_extra_fn += 1
                        idf1_extra_fp += 1
                    else:
                        # recovered track
                        del id_switched[gt_id]
                    
                # Check for identity switch or fragmentation
                if gt_id in id_mapping and id_mapping[gt_id] != pred_id:
                    # id switch
                    total_mismatches += 1
                    if gt_id not in id_switched:
                      id_switched[gt_id] = id_mapping[gt_id]
                id_mapping[gt_id] = pred_id

        # Calculate false positives and misses
        total_false_positives += len(pred_bboxes) - len(matches)
        idtp += len(matches)
        total_misses += len(gt_bboxes) - len(matches)
        # Calculate MOTP
        avg_motp = frame_tot_iou / len(gt_bboxes) if len(gt_bboxes) > 0 else None
        frame_avg_motp.append(avg_motp) 
        tot_iou += frame_tot_iou

    mota = 1 - (total_misses + total_false_positives + total_mismatches) / total_ground_truth
    motp = tot_iou / total_ground_truth

    # Calculate IDF1
    idfn = total_misses + idf1_extra_fn
    idfp = total_false_positives + idf1_extra_fp
    idf1 = idtp / (idtp + 0.5 * idfn + 0.5 * idfp)

    return mota, motp, idf1, frame_avg_motp

def plot_performance_graph(aligned_annotations, motp_per_frame, video_name):
    """
    Plots number of ground truth tracks vs number of predicted tracks for each frame,
    along with the MOTP for frames where it's available.
    """
    gt_bbox_xyxys = aligned_annotations['gt_bbox_xyxys']
    pred_bbox_xyxys = aligned_annotations['pred_bbox_xyxys']

    gt_bbox_xyxys_count = [len(x) for x in gt_bbox_xyxys]
    pred_bbox_xyxys_count = [len(x) for x in pred_bbox_xyxys]

    # Filter out None values from MOTP list and get corresponding frame numbers
    motp_values = [motp for motp in motp_per_frame if motp is not None]
    motp_frames = [i for i, motp in enumerate(motp_per_frame) if motp is not None]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f'Performance for {video_name}')
    ax.plot(gt_bbox_xyxys_count, label='Ground Truth', linestyle='-', marker='o', alpha=0.5)
    ax.plot(pred_bbox_xyxys_count, label='Predictions', linestyle='-', marker='o', alpha=0.5)
    # ax.plot(motp_frames, motp_values, label='Frame-Avg MOTP', linestyle='-', marker='o', alpha=0.5)

    ax.set_xlabel('Frame Number')
    ax.set_ylabel('Number of Tracks / MOTP')
    ax.legend()

    return fig

def get_torch_device():
    """
    Returns the device to be used for training and inference.
    """
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    else:
        return "CPU"