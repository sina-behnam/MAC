from abc import ABC, abstractmethod
from ultralytics import YOLO
import os
import shutil
from pathlib import Path
import subprocess

class Annotator(ABC):
    """
    Abstract base class for annotation tools.
    Provides functionality to annotate images and save the annotations in YOLO format.
    """

    @abstractmethod
    def annotate_image(self, image_path, output_path):
        """Annotate a single image."""
        pass
    
    @abstractmethod
    def annotate_dir(self, input_dir, output_dir):
        """Annotate all images in a directory."""
        pass
    
    @classmethod
    def choose_annotator(cls, method, model=None, **kwargs):
        """
        Choose the appropriate annotator based on the annotation method.

        :param method: Annotation method ('manual' or 'automated').
        :param model: Model path for automated annotation.
        :return: An instance of the selected Annotator subclass.
        """
        if method == "manual":
            return ManualAnnotator()
        elif method == "automated":
            return AutomatedAnnotator(model, **kwargs)
        else:
            raise ValueError("Invalid annotation method. Choose 'manual' or 'automated'.")

    @classmethod
    def annotate(cls, images, output_path, method="manual", model=None, **kwargs):
        """Annotate images using the appropriate annotator."""
        annotator = cls.choose_annotator(method, model, **kwargs)
        if isinstance(images, list):
            return annotator.annotate_dir(images, output_path)
        else:
            return annotator.annotate_image(images, output_path)


class ManualAnnotator(Annotator):
    """
    Manual annotation tool for annotating images with bounding boxes.
    """

    def annotate_image(self, image_path, output_path):
        """
        Annotate an image manually using the `labelImg` tool and save the annotations in YOLO format.

        :param image_path: Path to the image to be annotated.
        :param output_path: Path to save the annotations in YOLO format.
        """
        pass
        

    def annotate_dir(self, input_dir : list, output_dir):
        """
        Annotate all images in a directory manually using the `labelImg` tool and save the annotations in YOLO format.

        :param input_dir: Path to the directory containing the images to be annotated.
        :param output_dir: Path to save the annotations in YOLO format.
        """
        # Create the output directory if it does not exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # get input_dir base path 
        input_dir = os.path.dirname(input_dir[0])
        
        # Command to run labelImg
        command = ["labelImg", input_dir , output_dir]
        print(f"Running command: {' '.join(command)}")

        # Launch labelImg as a subprocess
        process = subprocess.Popen(command)

        # Wait for the user to close the labelImg window
        print("Please complete the annotation using labelImg. Waiting for you to close the tool...")
        process.wait()  # This will block until the labelImg window is closed

        print("labelImg window closed.")

        # Check the annotation directory for new files
        annotation_files = [f for f in os.listdir(output_dir) if f.endswith(".txt")]
        return annotation_files
        


class AutomatedAnnotator(Annotator):
    """
    Automated annotation tool for annotating images with bounding boxes using a pre-trained model.
    """

    def __init__(self, model=None, **kwargs):
        """
        Initialize the automated annotator with a pre-trained model.
        """
        super().__init__()
        try:
            self.yolo = YOLO(model)
        except:
            raise ValueError("Invalid model path. Please provide a valid path to a pre-trained YOLO model.")
        
        for k, v in kwargs.items():
            setattr(self, k, v)
        

    def annotate_image(self, image_path, output_path):
        """
        Annotate an image automatically using a pre-trained model and save the annotations in YOLO format.

        :param image_path: Path to the image to be annotated.
        :param output_path: Path to save the annotations in YOLO format.
        :param device: Device to run the model on ('cpu' or 'cuda').
        """
        results = self.yolo.predict(source=image_path,
                                    conf=float(self.conf) if hasattr(self, 'conf') else 0.25,
                                    device=self.device if hasattr(self, 'device') else 'cpu',
                                    save=False,
                                    )
        
        with open(output_path, 'w') as f:
            for r in results:
                for box,cls in zip(r.boxes.xywhn,r.boxes.cls):  # xywh format
                    cls = int(cls)  # Class index
                    x, y, w, h = box.tolist()  # Bounding box coordinates
                    # save with 2 floating point precision
                    f.write(f"{cls} {x:.2f} {y:.2f} {w:.2f} {h:.2f}\n")
        
            
    def annotate_dir(self, input_dir : list, output_dir):
        """
        Annotate all images in a directory automatically using a pre-trained model and save the annotations in YOLO format.

        :param input_dir: Path to the directory containing the images to be annotated.
        :param output_dir: Path to save the annotations in YOLO format.
        """
        # Create the output directory if it does not exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for image_file in input_dir:
            if image_file.endswith(".jpg"):
                image_path = image_file
                output_path = os.path.join(output_dir, os.path.basename(image_file).replace(".jpg", ".txt"))
                self.annotate_image(image_path, output_path)

        
        print(f"Automated annotation of images in {input_dir} completed.")
        print(f"Annotations saved to {output_dir}")


