import cv2

from image_processors.ImageProcessor import ImageProcessor


class Thresholder(ImageProcessor):
    def __init__(self):
        super().__init__()
    
    def __call__(self, orangeness:cv2.Mat) -> cv2.Mat:
        """Applies a threshold on the input image.

        Args:
            orangeness (cv2.Mat): Input image.

        Returns:
            cv2.Mat: A black and white image.
        """
        pass