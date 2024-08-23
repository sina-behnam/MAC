import cv2
import os
import argparse

def extract_frames(video_path, output_folder, frame_interval=30):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = video.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = f"frame_{saved_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    video.release()
    print(f"Extracted {saved_count} frames")






