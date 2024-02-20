from typing import Sequence, Tuple, Union
import cv2
import numpy as np
from image_processors.edge_filters.EdgeFilter import EdgeFilter
from image_processors.shape_postprocessors.ShapePostProcessor import ShapePostProcessor
from utils import compactRect, findOverlappingRotatedRectangles, minAreaRectRotatedRects


class RectMergePostProcess(ShapePostProcessor):
    def __init__(self, grow:float):
        """Rotated rectagles merging behavior.
        This behavior grows the rotated rectangles dimensions and attempts to find overlapping rectangles.

        Args:
            grow (float): Scaling factor to apply to existing rectangles.
        """
        super().__init__()
        self.grow = grow
    
    def __call__(self, img:cv2.Mat, rects:Union[np.ndarray[np.float_],None], valid:Union[np.ndarray[np.bool_], None], cnts:Sequence[cv2.Mat]) -> Tuple[np.ndarray[np.bool_],Union[np.ndarray[np.float_],None],Union[np.ndarray[np.float_],None],Union[np.ndarray[np.float_],None], Sequence[cv2.Mat]]:
        if valid is None or not np.any(valid):
            return valid, None, None, None, cnts
        r = rects.copy()
        
        # Get pairs of rectangles
        newRects = np.array([minAreaRectRotatedRects(pair) for pair in findOverlappingRotatedRectangles(img, r, valid, self.grow)], ndmin=2)
        if self.debug and newRects.shape[-1] != 0:
            dbg = img.copy()
            for rr in newRects:
                dbg = cv2.drawContours(dbg, [np.int_(cv2.boxPoints(compactRect(rr)))], 0, (0,255,0), 2)
                
            for rr in r:
                dbg = cv2.drawContours(dbg, [np.int_(cv2.boxPoints(compactRect(rr)))], 0, (255,0,0), 1)
                
            self.debugImg(dbg)
        
        # Get areas of the new rectangles
        newRectareas = newRects[:,2:4].prod(1)
        if len(newRects) != 1:
            newBoxes = np.array([np.int_(cv2.boxPoints(compactRect(rr))) for rr in newRects])
        else:
            newBoxes = np.empty((1,0), np.float_)
        return valid, newRects, newRectareas, newBoxes, cnts