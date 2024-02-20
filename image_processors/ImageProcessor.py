import time
import cv2
import numpy as np


class ImageProcessor:
    def __init__(self):
        self.debug = False
    
    @property
    def notifiedDebug(self) -> bool:
        return self.debug
    
    @notifiedDebug.setter
    def notifiedDebug(self, value:bool):
        if not value:
            self.destroyWindow()
        self.debug = value
    
    def __call__(self, *args, **kwargs):
        pass
    
    def debugImg(self, img: cv2.Mat):
        """Debug an image witht class name as the title.

        Args:
            img (cv2.Mat): Image to display.
        """
        if self.debug:
            cv2.imshow(self.__class__.__name__, img)
    
    def destroyWindow(self):
        """Destroy the window created by self.debugImg.
        """
        if self.debug:
            cv2.destroyWindow(self.__class__.__name__)
    
    def speedBenchmark(self, iterations:int, *args, **kwargs) -> np.ndarray[np.float_]:
        """Runs the behavior multiple times and gets the time spent.

        Args:
            iterations (int): Number of iterations to perform.

        Returns:
            np.ndarray[np.float_]: Time spent at each iteration in seconds.
        """
        res = np.zeros((iterations), np.float_)
        for i in range(iterations):
            start = time.time()
            self.__call__(*args, **kwargs)
            end = time.time()
            res[i] = end-start
        return res