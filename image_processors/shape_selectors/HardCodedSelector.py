

from typing import Dict, List, Tuple
from cv2 import Mat
from numpy import float_, ndarray
from image_processors.shape_selectors.ShapeSeletor import ShapeSelector


class HardCodedSelector(ShapeSelector):
    def __init__(self):
        super().__init__()
        
    def __call__(self, img: Mat, scores: List[Dict[str, float]], warps: List[Mat], trapezoids: ndarray[float_]) -> Tuple[List[int], List[ndarray[float_]]]:
        pass
    
    