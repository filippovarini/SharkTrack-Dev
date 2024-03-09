import os
import cv2


def extract_frames(video_path, frame_output_directory, source, desired_fps, max_frames=1000):
    if not os.path.exists(frame_output_directory):
      os.makedirs(frame_output_directory)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")

    interval = 1 / desired_fps
    next_capture_time = 0.0

    saved_frames_cnt = 0

    while cap.isOpened() and saved_frames_cnt < max_frames:
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if current_time >= next_capture_time:
            filename =  os.path.join(frame_output_directory, f"{source}_frame{saved_frames_cnt}.jpg")
            cv2.imwrite(filename, frame)
            saved_frames_cnt += 1
            next_capture_time += interval
        
    cap.release()
    print(f"Extracted {saved_frames_cnt} frames from {video_path}")
