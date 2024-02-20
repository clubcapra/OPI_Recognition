from typing import Sequence, Tuple, Union
import cv2
import numpy as np
from image_processors.edge_detectors.EdgeDetector import EdgeDetector
from image_processors.edge_filters.EdgeFilter import EdgeFilter
from image_processors.shape_postprocessors.ShapePostProcessor import ShapePostProcessor
from utils import compactRect


class FillPostProcess(ShapePostProcessor):
    def __init__(self, edgeDetector:EdgeDetector, edgeFilter:EdgeFilter, grow:float):
        """Graphical rectangle merging behavior.
        This behavior uses applies a scale on the input rectangles and then creates a new image with those areas filled.
        Then it uses an EdgeDetector to find the contours of the merged shapes.
        Finally, an EdgeFilter is used to get the resulting merged rotated rectangles.

        Args:
            edgeDetector (EdgeDetector): Edge detector to use to find the merged rectangles.
            edgeFilter (EdgeFilter): Edge filter to filter the new edges found.
            grow (float): Scale of the rectangles to fill.
        """
        super().__init__()
        self.edgeDetector = edgeDetector
        self.edgeFilter = edgeFilter
        self.grow = grow
    
    def __call__(self, img:cv2.Mat, rects:Union[np.ndarray[np.float_],None], valid:Union[np.ndarray[np.bool_], None], cnts:Sequence[cv2.Mat]) -> Tuple[np.ndarray[np.bool_],Union[np.ndarray[np.float_],None],Union[np.ndarray[np.float_],None],Union[np.ndarray[np.float_],None], Sequence[cv2.Mat]]:
        if valid is None or not np.any(valid):
            return valid, None, None, None, cnts
        h, w, _ = img.shape
        mask = np.zeros((h,w), np.uint8)
        for r in rects[valid]:
            (xx,yy),(ww,hh),rr = compactRect(r)
            
            
            # Expand the boxes to allow 2 seperated shapes to become one
            expanded = cv2.boxPoints(((xx,yy),(ww*self.grow, hh*self.grow),rr))
            expanded = np.int_(expanded)
            mask = cv2.drawContours(mask, [expanded], 0, (255), -1)
            
        # Get the new contours
        cnts2 = self.edgeDetector(mask)
        _, newRects, newRectAreas, newBoxes = self.edgeFilter(img, cnts2)
        return valid, newRects, newRectAreas, newBoxes, cnts