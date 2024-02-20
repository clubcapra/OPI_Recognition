from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
from image_processors.ImageProcessor import ImageProcessor
from utils import debugScore


class ShapeIdentifier(ImageProcessor):
    def __init__(self):
        super().__init__()
        
    def debugScore(self, score:Dict[str,float], ndigits=2):
        """Debugs the scores provided.

        Args:
            score (Dict[str,float]): A dictionnary containing the name of each score and their respective value.
            ndigits (int, optional): Number of digits to show after the period. Defaults to 2.
        """
        if not self.debug:
            return
        debugScore(score, ndigits)
        
    def __call__(self, img: cv2.Mat, warps: Union[List[cv2.Mat], None], rectAreas:Union[np.ndarray[np.float_], None]) -> Tuple[Union[List[Dict[str, float]], None], Union[List[cv2.Mat], None]]:
        """Identifies the features of shapes found in each warps and attributes a score to each feature.

        Args:
            img (cv2.Mat): Input image (used for debugging).
            warps (Union[List[cv2.Mat], None]): Rectified selected areas od the base image.
            rectAreas (Union[np.ndarray[np.float_], None]): Areas of the rectangles per warps.

        Returns:
            Tuple[Union[List[Dict[str, float]], None], Union[List[cv2.Mat], None]]: (scores, warps)
            scores (Union[List[Dict[str, float]], None]): None if warps is None. A list of the scores found for each of the resulting warps.
            warps (Union[List[cv2.Mat], None]): None if warps is None. A list of processed warps. The size of this list may differ from the size of the input warps' list.
        """
        pass
