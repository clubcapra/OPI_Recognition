import cv2

from image_processors.ImageProcessor import ImageProcessor



class Normalizer(ImageProcessor):
    def __init__(self):
        super().__init__()
        
    def __call__(self, img:cv2.Mat) -> cv2.Mat:
        """Normalize image.

        Args:
            img (cv2.Mat): Input image.

        Returns:
            cv2.Mat: A normalized version of the input image.
        """
        pass