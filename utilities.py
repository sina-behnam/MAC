import cv2
import os

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


def split_and_extract_frames(video_path, output_folder, frame_interval=30, split_ratio=0.5):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the split point based on the ratio
    split_frame = int(total_frames * split_ratio)

    # Define output paths for the video halves
    part1_video_path = os.path.join(output_folder, "part1_video.mp4")
    part2_video_path = os.path.join(output_folder, "part2_video.mp4")

    # Save the first part of the video
    save_video_section(video, 0, split_frame, part1_video_path, fps, width, height)
    # Save the second part of the video
    save_video_section(video, split_frame, total_frames, part2_video_path, fps, width, height)

    # Extract frames for the first half
    extract_frames_from_section(video_path, 0, split_frame, output_folder + "_part1", frame_interval)
    # Extract frames for the second half
    extract_frames_from_section(video_path, split_frame, total_frames, output_folder + "_part2", frame_interval)

    video.release()

def save_video_section(video, start_frame, end_frame, output_path, fps, width, height):
    # Set the video writer with output path and codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Set the starting frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Write frames to the output video file
    for frame_idx in range(start_frame, end_frame):
        ret, frame = video.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    print(f"Saved video section to {output_path}")

def extract_frames_from_section(video_path, start_frame, end_frame, output_folder, frame_interval):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frame_count = start_frame
    saved_count = 0

    while frame_count < end_frame:
        ret, frame = video.read()
        if not ret:
            break

        if (frame_count - start_frame) % frame_interval == 0:
            frame_filename = f"frame_{saved_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    video.release()
    print(f"Extracted {saved_count} frames from {output_folder}")

# # Example usage
# video_path = "data/seafloor_footage.mp4"
# output_folder = "data/outputs"
# frame_interval = 30
# split_ratio = 0.5  # Split the video in half; can be changed (e.g., 0.3 for 30/70 split)

# split_and_extract_frames(video_path, output_folder, frame_interval, split_ratio)




