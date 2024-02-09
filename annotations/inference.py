from annotations.viame import track_history_to_viame
import pandas as pd
import os

def track_history_to_csv(track_history, video_name, output_folder, track_fps):
    assert 'pred_bbox_xyxys' in track_history and 'pred_confidences' in track_history and 'pred_track_ids' in track_history, 'Invalid track history'
    assert len(track_history['pred_bbox_xyxys']) == len(track_history['pred_confidences']) == len(track_history['pred_track_ids']), 'Lengths do not match'

    frames_data = []

    for i in range(len(track_history['pred_bbox_xyxys'])):
        frame_data = {
            'bbox_xyxys': track_history['pred_bbox_xyxys'][i],
            'confidences': track_history['pred_confidences'][i],
            'track_ids': track_history['pred_track_ids'][i],
            'frame_id': i,
        }
        frames_data.append(frame_data)
    
    all_rows = []

    for frame_data in frames_data:
        assert len(frame_data['bbox_xyxys']) == len(frame_data['confidences']) == len(frame_data['track_ids']), f'Frame detection lengths do not match: {frame_data}'

        num_detections = len(frame_data['bbox_xyxys'])
        for i in range(num_detections):
            # For each detection, create a row and append to `all_rows`
            row = {
                'track_id': frame_data['track_ids'][i],
                'frame_id': frame_data['frame_id'],
                'confidence': frame_data['confidences'][i],
                'bbox_xyxy': frame_data['bbox_xyxys'][i],
            }
            all_rows.append(row)

    # Convert the list of rows to a DataFrame
    df = pd.DataFrame(all_rows)

    viame_df = track_history_to_viame(df, track_fps)

    # Save the DataFrame to a CSV file
    csv_file_path = os.path.join(output_folder, f"{video_name}_track_history.csv")
    viame_df.to_csv(csv_file_path, index=False)

    print(f"Data saved to {csv_file_path}.")