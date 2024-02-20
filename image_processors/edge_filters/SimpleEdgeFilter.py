from typing import Sequence, Tuple, Union
import cv2
import numpy as np
from image_processors.edge_filters.EdgeFilter import EdgeFilter
from utils import compactRect, expandRect


class SimpleEdgeFilter(EdgeFilter):
    def __init__(self, minArea:float, maxArea:float, margin:float):
        """Base implementation of an edge filtering behavior.

        Args:
            minArea (float): Minimum area ratio (0 - 1) to keep.
            maxArea (float): Maximum area ration (0 - 1) to keep.
            margin (float): Margin ratio (0 - 1) heightwise to ignore from the borders of the image.
        """
        super().__init__()
        self.minArea = minArea
        self.maxArea = maxArea
        self.margin = margin
        
    def __call__(self, img:cv2.Mat, cnts:Sequence[cv2.Mat]) -> Tuple[np.ndarray[np.bool_],Union[np.ndarray[np.float_], None],Union[np.ndarray[np.float_], None],Union[np.ndarray[np.float_], None]]:
        height, width, _ = img.shape
        minArea = self.minArea * height * width
        maxArea = self.maxArea * height * width
        margin = self.margin * height
        
        # Getting the area of each contour
        areas = np.array([cv2.contourArea(contour) for contour in cnts])
        valid = (areas > minArea) & (areas < maxArea)
        
        # If no valid left, return
        if not np.any(valid):
            return valid, None, None, None
        
        # Get the area of each rectangle (a contour could have an area way smaller than a rectangle)
        rects = np.array([expandRect(cv2.minAreaRect(c)) if v else (0,0,0,0,0)  for v, c in zip(valid, cnts)])
        rectAreas = rects[:,2] * rects[:,3]
        valid &= rectAreas < maxArea
        
        # If no valid left, return
        if not np.any(valid):
            return valid, None, None, None
        
        # Get the box points
        boxes = np.array([cv2.boxPoints(compactRect(r)) if v else np.zeros((4,2),np.float_) for v, r in zip(valid, rects)])
        
        valid &= np.any((boxes > (margin, margin)) & (boxes < (width-margin, height-margin)), (1,2))
        
        return valid, rects, rectAreas, boxes