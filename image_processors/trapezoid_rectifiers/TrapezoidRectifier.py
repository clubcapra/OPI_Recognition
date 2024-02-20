from typing import List, Union
import cv2
import numpy as np
from image_processors.ImageProcessor import ImageProcessor


class TrapezoidRectifier(ImageProcessor):
    def __init__(self):
        
        super().__init__()
        
    def __call__(self, img: cv2.Mat, trapezoids:Union[np.ndarray[np.float_], None]) -> Union[List[cv2.Mat], None]:
        """Given four corners (trapezoids) and an image, warp the selected area to be squared up, resulting in a perfect rectangle.
        This does not account for lens distorsion.

        Args:
            img (cv2.Mat): Input image from wich to grab the area.
            trapezoids (Union[np.ndarray[np.float_], None]): Array of coordinates with shape (n, 4, 2).

        Returns:
            Union[List[cv2.Mat], None]: None if no trapezoids. Rectified images of length n.
        """
        pass