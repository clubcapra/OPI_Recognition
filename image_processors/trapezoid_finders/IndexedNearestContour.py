from itertools import chain
from typing import Sequence, Union
import cv2
import numpy as np
from image_processors.trapezoid_finders.TrapezoidFinder import TrapezoidFinder


class IndexedNearestContour(TrapezoidFinder):
    def __init__(self):
        """Finds the trapezoids by getting the clossest valid contours from each rectangle's corners.
        The main difference with NearestContour is that only contours used to create the rect will be used, reducing the amount of checks to perform.
        """
        super().__init__()
        
    def __call__(self, img:cv2.Mat, cnts: Sequence[cv2.Mat], valid: Union[np.ndarray[np.bool_], None], rects: Union[np.ndarray[np.float_], None], boxes:Union[np.ndarray[np.float_], None]) -> Union[np.ndarray[np.float_], None]:
        if valid is None or not np.any(valid):
            return None
        res = None
        
        candidates = np.array([vvv for vvv in chain(*[values for values in [vv[1] for vv in filter(lambda c: c[0], zip(valid, cnts))]])]).squeeze()
        trapezoids = []
        if boxes is None:
            return None
        for b, c in zip(boxes, cnts):
            # Draw the rough outline
            if self.debug:
                res = cv2.drawContours(img.copy(), [np.int_(b)], 0, (0,0,255),2)
            
            # Find the trapezoid contained in this expanded shape
            trapezoid = np.zeros((4,2), np.int_)
            
            # Get the clossest contour point for each corner
            for i, corner in enumerate(b):
                # Get all distances
                m = np.linalg.norm(c - corner, axis=0, keepdims=True)
                # Find minimum
                idx = np.where(m == m.min())
                # Grab minimum's contour coordinates
                if idx[0].size > 1:
                    trapezoid[i] = candidates[idx[0][0]]
                else:
                    trapezoid[i] = candidates[idx[0]]
            
            # Draw the resulting trapezoid
            if self.debug:
                res = cv2.drawContours(res, [trapezoid], 0, (255,255,0),3)
            trapezoids.append(trapezoid)
            
        if self.debug and res is not None:
            cv2.imshow('NearestContour', res)
        
        return np.array(trapezoids)