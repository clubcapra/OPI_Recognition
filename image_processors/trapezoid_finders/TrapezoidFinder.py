from typing import Sequence, Union
import cv2
import numpy as np
from image_processors.ImageProcessor import ImageProcessor


class TrapezoidFinder(ImageProcessor):
    def __init__(self):
        super().__init__()
        
    def __call__(self, img:cv2.Mat, cnts: Sequence[cv2.Mat], valid: Union[np.ndarray[np.bool_], None], rects: Union[np.ndarray[np.float_], None], boxes:Union[np.ndarray[np.float_], None]) -> Union[np.ndarray[np.float_], None]:
        """Finds the trapezoid that englobes each detected shape.

        Args:
            img (cv2.Mat): Input image (used for debugging).
            cnts (Sequence[cv2.Mat]): Sequence of contours found by an EdgeDetector with length n.
            valid (Union[np.ndarray[np.bool_], None]): Array of bool with shape (n) representing which input values are valid.
            rects (Union[np.ndarray[np.float_], None]): Array of rects with shape (n, 5) found by an EdgeFilter having the format (x, y, width, height, rotation).
            boxes (Union[np.ndarray[np.float_], None]): Array of coordinates of the rects' corners with shape (n, 4, 2).

        Returns:
            Union[np.ndarray[np.float_], None]: None if no valids. Array of coordinates of the corners found. This array is of shape (valid.sum(), 4, 2).
        """
        pass