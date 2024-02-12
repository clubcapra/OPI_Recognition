

from typing import Dict, List, Tuple
import cv2
import numpy as np
from image_processors.shape_selectors.ShapeSeletor import ShapeSelector


class HardCodedSelector(ShapeSelector):
    def __init__(self):
        super().__init__()
        
    def __call__(self, img: cv2.Mat, scores: List[Dict[str, float]], warps: List[cv2.Mat], trapezoids: np.ndarray[np.float_]) -> Tuple[List[int], List[np.ndarray[np.float_]], List[float]]:
        pass
    
    