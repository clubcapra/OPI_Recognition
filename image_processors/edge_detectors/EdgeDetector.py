from typing import Sequence
import cv2

from image_processors.ImageProcessor import ImageProcessor


class EdgeDetector(ImageProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, threshold:cv2.Mat) -> Sequence[cv2.Mat]:
        """Gets the contours of shapes found in the input image.

        Args:
            threshold (cv2.Mat): A grayscale or black and white image to analyze.

        Returns:
            Sequence[cv2.Mat]: A sequence containing the contours of the shapes found.
        """
        pass    