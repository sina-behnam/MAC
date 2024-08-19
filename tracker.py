import os
import logging
import cv2
import argparse
from ultralytics import YOLO
from tqdm import tqdm

# Suppress OpenMP duplicate library warnings (if needed)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def export_model(model, model_path , device, img_size):
    """Exports the model to ONNX for CPU or TensorRT for GPU based on the device."""
    path = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).split(".")[0]
    if device == "cpu":
        export_path = os.path.join(path, model_name + ".onnx")
        model.export(format="onnx", half=False, device=device, imgsz=img_size)
        print(f"Model exported to ONNX format at {export_path}")
    elif device == "cuda":
        export_path = os.path.join(path, model_name + ".engine")
        model.export(format="engine", half=True, device=device, imgsz=img_size)
        print(f"Model exported to TensorRT format at {export_path}")
    return export_path

def main(args):
    # Load the YOLOv8 model
    model_path = args.model if args.model else "models/v4/best.pt"
    model = YOLO(model_path)

    # Determine the device based on the argument
    device = "cuda" if args.device == "gpu" else "cpu"

    # Open the video file
    video_path = args.video
    cap = cv2.VideoCapture(video_path)

    # Get the video frame width, height, and frames per second (fps)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Video Resolution: {frame_width}x{frame_height}")
    print(f"Frames per Second: {fps}")

    if args.export_model:
        # Export the model according to the device type and image size
        try:
            img_size = args.img_size if args.img_size else 640

            export_model_path = export_model(model, model_path , device, img_size)
            model = YOLO(export_model_path, task="detect", verbose=True)
        except Exception as e:
            print(f"Error loading the exported model: {e}")
    else:
        model = YOLO(model_path, task="detect", verbose=True)

    if args.export:
        # Define the codec and create a VideoWriter object to save the output video
        output_path = args.output if args.output else "data/output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can use 'XVID', 'MJPG', or others
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Disable logging output from the YOLOv8 model
    logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

    # Create a progress bar using tqdm
    with tqdm(total=total_frames, desc="Processing Video", unit="frame") as pbar:
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=True, show_labels=False, show=False, conf=args.confidence, tracker=args.tracker)

                # Count the number of objects detected
                object_count = len(results[0].boxes)

                # Extract boxes from results and draw them manually
                for result in results[0].boxes:
                    # Extract bounding box coordinates (x1, y1, x2, y2)
                    x1, y1, x2, y2 = map(int, result.xyxy[0].tolist())
                    
                    # Draw a thin bounding box without labels
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=8)

                # Add object count text on top of the frame
                text = f"Object Count: {object_count}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                color = (255, 0, 0)  # Blue text
                thickness = 4
                position = (50, 50)  # Position the text at the top-left corner
                
                # Put the text on the frame
                cv2.putText(frame, text, position, font, font_scale, color, thickness)

                if args.export:
                    # Write the annotated frame to the output video
                    out.write(frame)

                if args.display:
                    # Display the frame with bounding boxes
                    cv2.imshow("Tracking", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Update the progress bar
                pbar.update(1)
            else:
                # Break the loop if the end of the video is reached
                break

    # Release the video capture and writer objects, and close any open windows
    cap.release()

    if args.export:
        out.release()
        
    if args.display:
        cv2.destroyAllWindows()

    if args.export:
        print(f"Video with bounding boxes saved to {args.output}")

if __name__ == "__main__":
    # Argument parser to handle command-line arguments
    parser = argparse.ArgumentParser(description="YOLOv8 Object Tracking in Video")

    parser.add_argument(
        "--video", 
        type=str, 
        required=True, 
        help="Path to the input video file."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None, 
        help="Path to the YOLOv8 model file. If not specified, a default model is used."
    )
    parser.add_argument(
        "--export_model",
        action="store_true",
        default=False,
        help="Export the model to ONNX for CPU or TensorRT for GPU."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cpu", 
        choices=["cpu", "gpu"], 
        help="Device to use for model inference. Choose 'cpu' or 'gpu'."
    )
    parser.add_argument(
        "--export", 
        action="store_true", 
        help="Export the tracking results as a video file."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/output_video.mp4", 
        help="Path to save the output video file (used only if --export is specified)."
    )
    parser.add_argument(
        "--display", 
        action="store_true", 
        help="Display the tracking results in a window."
    )
    parser.add_argument(
        "--confidence", 
        type=float, 
        default=0.3, 
        help="Confidence threshold for object detection."
    )
    parser.add_argument(
        "--tracker", 
        type=str, 
        default="custom_tracker.yaml", 
        help="Path to the tracker configuration file."
    )
    parser.add_argument(
        "--img_size", 
        type=int, 
        nargs=2, 
        default=None, 
        help="The Image size that it has been trained on. By default, it is 640. The format is (width, height)."
    )

    args = parser.parse_args()

    # Run the main function with parsed arguments
    main(args)