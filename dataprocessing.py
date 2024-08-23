from utilities import *
from annotator import Annotator
import argparse
import glob


def arguments():
    # Frame extraction arguments
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file and annotate images with bounding boxes. The annotations can be done manually or automatically using a YOLOv8 pre-trained model. \
            In case of using a custom model that is not in a YOLOv8 format, you may need to implement your annotation class method by inheriting from the Annotator class."
    )
    parser.add_argument('--extract_frames', action='store_true', help='Extract frames from video')
    parser.add_argument('--video_path', type=str, help='Path to the video file')
    parser.add_argument('--output_folder', type=str, help='Path to the output folder')
    parser.add_argument('--frame_interval', type=int, default=30, help='Interval to extract frames')
    # Annotator arguments
    parser.add_argument('--annotate', action='store_true', help='Annotate images')
    parser.add_argument('--images', type=str, help='Path to the images')
    parser.add_argument('--output_path', type=str, help='Path to save the annotations')
    parser.add_argument('--method', type=str, default="manual", choices=["manual", "automated"], help="Annotation method it is advisable to use `labelImg` for manual annotation and used this script for automated annotation")
    parser.add_argument('--model', type=str, help='Path to the YOLOv8 model')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], 
        help="Device to use for model prediction. Choose 'cpu' or 'gpu'."
    )

    return parser.parse_args()
    

def main():
    
    args = arguments()
    
    if args.extract_frames:
        extract_frames(args.video_path, args.output_folder, args.frame_interval)

    if args.annotate:
        Annotator.annotate(
            images=glob.glob(args.images),
            output_path=args.output_path,
            method=args.method,
            model=args.model,
            conf=args.conf,
            device=args.device
        )

if __name__ == "__main__":
    main()


