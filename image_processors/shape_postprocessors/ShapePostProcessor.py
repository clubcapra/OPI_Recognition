from typing import Sequence, Tuple, Union
import cv2
import numpy as np
from image_processors.ImageProcessor import ImageProcessor


class ShapePostProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()
    
    def __call__(self, img:cv2.Mat, rects:Union[np.ndarray[np.float_],None], valid:Union[np.ndarray[np.bool_], None], cnts:Sequence[cv2.Mat]) -> Tuple[np.ndarray[np.bool_],Union[np.ndarray[np.float_],None],Union[np.ndarray[np.float_],None],Union[np.ndarray[np.float_],None], Sequence[cv2.Mat]]:
        """Post-process shapes. Most of the time, the top and bottom of the ERICards are detected as separate shapes.
        This behavior is used to find those cases and merger the halves if required.
        
        Where 'n' represents the number of elements in the input arrays.
        Where 'm' represents the number of elements in SOME output arrays.

        Args:
            img (cv2.Mat): Input image (used for debugging).
            rects (np.ndarray[np.float_]): Array of rectangles with shape (n, 5) in the format (x, y, width, height, angle).
            valid (np.ndarray[np.bool_]): Array of bool with shape (n) representing the rects to use for post-processing.
            cnts (Sequence[cv2.Mat]): Sequence of contours as returned by an EdgeDetector.

        Returns:
            Tuple[np.ndarray[np.bool_],np.ndarray[np.float_],np.ndarray[np.float_],np.ndarray[np.float_], Sequence[cv2.Mat]]: (updatedValid, newRects, newRectAreas, newBoxes, cnts)
                Dimensions 0 of some arrays may differ from input arrays. This is because the merging process modifies the amount of total shapes.
                
            updatedValid (np.ndarray[np.bool_]): Array of bool with shape (n) representing selected rectangles. 
                When merging occurs with rects[i] and rects[j]: 
                    if valid[i] and valid[j] are true and their merged result is valid, 
                    updatedValid[i] and updatedValid[j] will be set to true.
            newRects (np.ndarray[np.float_]): Array of the merged rectangles with shape (m, 5) in the format (x, y, width, height, angle).
            newRectAreas (np.ndarray[np.float_]): Array of the merged rectangles' width with shape (m).
            newBoxes (np.ndarray[np.float_]): Array of the coordinates of the corners for the new rectangles with shape (m, 4, 2).
            cnts (Sequence[cv2.Mat]): Updated sequence of contours to use going forward.
        """
        pass