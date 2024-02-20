from typing import Sequence, Tuple, Union
import cv2
import numpy as np
from image_processors.ImageProcessor import ImageProcessor


class EdgeFilter(ImageProcessor):
    def __init__(self):
        super().__init__()
        
    def __call__(self, img:cv2.Mat, cnts:Sequence[cv2.Mat]) -> Tuple[np.ndarray[np.bool_],Union[np.ndarray[np.float_], None],Union[np.ndarray[np.float_], None],Union[np.ndarray[np.float_], None]]:
        """Filters the input contours.

        Args:
            img (cv2.Mat): Input image (used for debugging).
            cnts (Sequence[cv2.Mat]): Sequence of contours with length n as returned by an EdgeDetector.

        Returns:
            Tuple[np.ndarray[np.bool_],Union[np.ndarray[np.float_], None],Union[np.ndarray[np.float_], None],Union[np.ndarray[np.float_], None]]: (valid, rects, rectAreas, boxes)
                All arrays have matching dimension 0.
            valid (np.ndarray[np.bool_]): Array of bools of shape (n) corresponding to valid results.
            rects (Union[np.ndarray[np.float_], None]): None if there are no valid. Array of rectangles of shape (n, 5) where elements are of form (x, y, width, height, angle).
            rectAreas (Union[np.ndarray[np.float_], None]): None if there are no valid. Array of the areas of the rectangles with shape (n).
            boxes (Union[np.ndarray[np.float_], None]): None if there are no valid. Array of shape (n, 4, 2) representing the coordinates of the corners of the rectangles.
        """
        pass