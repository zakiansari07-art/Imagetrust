import cv2
from pathlib import Path
from PIL import Image
import numpy as np

class ImageForensicsAnalyzer:

    def analyze(self, image_path: str):

        path = Path(image_path)
        if not  path.exists():
            raise FileNotFoundError (f"Image not found: {image_path}")

        with Image.open(path) as image:
            image_format = image.format
            height, width = image.size
            original_mode = image.mode
            has_exif = len(image.getexif()) > 0
            rgb_image = image.convert("RGB")

        grayscale_image = cv2.cvtColor(np.array(rgb_image), cv2.COLOR_RGB2GRAY)

        sharpness_score =  cv2.Laplacian(grayscale_image, cv2.CV_64F).var()

        average_brightness = grayscale_image.mean()

        return {
            "file_name": path.name,
            "file_format": image_format,
            "width": width,
            "height": height,
            "original_color_mode": original_mode,
            "has_exif_metadata": has_exif,
            "sharpness_score": round(float(sharpness_score), 2),
            "average_brightness": round(float(average_brightness), 2)
        }
            