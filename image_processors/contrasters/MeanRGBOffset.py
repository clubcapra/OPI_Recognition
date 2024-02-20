from typing import List, Tuple, Union
import cv2
import numpy as np
from common import ORANGE
from image_processors.contrasters.Contraster import Contraster


class MeanRGBOffset(Contraster):
    def __init__(self, color:Union[cv2.Mat, np.ndarray, Tuple[int, int, int], List[int]]=ORANGE):
        """Gets the mean of the absolute offset from an RGB color.

        Args:
            color (Union[cv2.Mat, np.ndarray, Tuple[int, int, int], List[int]], optional): A color in the format BGR. Defaults to ORANGE.
        """
        super().__init__()
        self.color = np.int16(color)
        
    def __call__(self, img: cv2.Mat) -> cv2.Mat:
        dist = np.abs(img.astype(np.int16) - ORANGE)
        res = dist.mean(2).astype(np.uint8)
        self.debugImg(res)
        return res