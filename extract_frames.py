import os
import cv2


def extract_frames(video_path, label, output_dir, frame_interval=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if frame_count % frame_interval == 0 and success:
            frame_file = os.path.join(output_dir, f"{label}_{frame_count}.jpg")
            cv2.imwrite(frame_file, frame)
        frame_count += 1

    cap.release()
