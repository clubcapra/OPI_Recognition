import cv2
import numpy as np
from image_processors.normalizers.Normalizer import Normalizer


class SimpleNormalizer(Normalizer):
    def __init__(self):
        """Base implementation of a Normalizer.
        Normalizes each color channel individually.
        """
        super().__init__()
        
    def __call__(self, img:cv2.Mat) -> cv2.Mat:
        # Get the min value per channel
        mn = img.min((0,1))
        # Get the max value for each channel
        mx = img.max((0,1))
        # Calculate delta
        dt = mx-mn
        
        # Get image as float
        fimg = img.astype(np.float32)
        # Scale up from 0-255 each channel (this probably isn't the best idea, but it seems to be working)
        res = ((fimg[:,:]-mn)/dt*255).astype(np.uint8)
        self.debugImg(res)
        return res