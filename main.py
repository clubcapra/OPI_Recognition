from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from cinput import cinput, ynValidator
from common import CONVERTED_PATH, KEY_ESC, KEY_LEFT, KEY_RIGHT
from image_processors import ImageProcessor, OPIFinder, contrasters, edge_detectors, edge_filters, normalizers, shape_identifiers, shape_postprocessors, shape_selectors, trapezoid_finders, trapezoid_rectifiers, thresholders

from utils import convert, ensureExists


class ImageBrowserBehavior:
    def __init__(self, imgs: List[cv2.Mat], behaviors:OPIFinder):
        self.imgs = imgs
        self.index = 0
        self.stepIndex = 0
        self.colors = []
        self.img = imgs[0]
        self.behaviors = behaviors
        
    def mouseEvent(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.colors.append(self.img[y,x])
            arr = np.array(self.colors)
            print(arr.mean(0))

    def loop(self):
        k = cv2.waitKey(10)
        if k == -1:
            return
        b = self.behaviors.getStepNumber(self.stepIndex)
        if k == KEY_LEFT:
            self.index = len(self.imgs) - 1 if self.index == 0 else self.index - 1
        if k == KEY_RIGHT:
            self.index = (self.index + 1) % len(self.imgs)
        # if k == KEY_DOWN:
        #     # if not b.debug:
        #     #     b.destroyWindow()
        #     while True:
        #         self.stepIndex = len(self.behaviors.choices) - 1 if self.stepIndex == 0 else self.stepIndex - 1
        #         b = self.behaviors.getStepNumber(self.stepIndex)
        #         if b is not None:
        #             break
        # if k == KEY_UP:
        #     # if not b.debug:
        #     #     b.destroyWindow()
        #     while True:
        #         self.stepIndex = (self.stepIndex + 1) % len(self.behaviors.choices)
        #         b = self.behaviors.getStepNumber(self.stepIndex)
        #         if b is not None:
        #             break
        # if k == ord(' '):
            # b.notifiedDebug ^= True
        if k == KEY_ESC:
            exit(0)
        
        # dbg = b.debug
        # b.debug = True
        self.behaviors.find2(self.imgs[self.index])
        # b.debug = dbg

if __name__ == "__main__":
    ensureExists()
    convert()
    
    imgs = [cv2.imread(str(p)) for p in CONVERTED_PATH.iterdir()]
    
    # ib = ImageBrowser(imgs)
    # ib.processingMethods.append(noProcess)
    # ib.processingMethods.append(normalize)
    # # ib.processingMethods.append(thresh)
    # # ib.processingMethods.append(gray)
    # ib.processingMethods.append(red)
    # ib.processingMethods.append(orangeness1)
    # ib.processingMethods.append(orangeness1Thresh)
    # # ib.processingMethods.append(orangeness2)
    # # ib.processingMethods.append(canny)
    # ib.processingMethods.append(filterThresh)
    # ib.processingMethods.append(straighten)
    # ib.processingMethods.append(ocr)
    # ib.processingMethods.append(final)
    
    finder = OPIFinder()
    finder.steps['normalize']['simple'] = normalizers.SimpleNormalizer()
    
    finder.steps['orangeness']['meanRGBOffset'] = contrasters.MeanRGBOffset()
    
    finder.steps['threshold']['simple'] = thresholders.SimpleThreshold(50)
    
    finder.steps['edgeDetect']['simple'] = edge_detectors.SimpleEdgeDetect()
    
    finder.steps['edgeFilter']['simple'] = edge_filters.SimpleEdgeFilter(0.008, 0.9, 0.001)
    
    finder.steps['postProcessShapes']['fillExpand'] = shape_postprocessors.FillPostProcess(finder.steps['edgeDetect']['simple'], finder.steps['edgeFilter']['simple'], 1.1)
    finder.steps['postProcessShapes']['rectMerge'] = shape_postprocessors.RectMergePostProcess(finder.steps['edgeFilter']['simple'], 1.1)
    
    finder.steps['findTrapezoids']['simple'] = trapezoid_finders.NearestContour()
    
    finder.steps['rectifyTrapezoids']['simple'] = trapezoid_rectifiers.BasicRectify()
    
    redContraster = contrasters.Red(normalizers.SimpleNormalizer())
    
    finder.steps['scoreShapes']['simple'] = shape_identifiers.BasicShapeIdentifier(0.035, 0.04, redContraster)
    
    # finder.steps['selectShapes']['logistic'] = LogisticRegressionSelector()
    # for t, tt in finder.steps.items():
    #     for p, pp in tt.items():
    #         pp.debug = True
    
    # finder.find(imgs[2])
    ovw = cinput("Overwite? (y/n)", str, ynValidator) == 'y'
    test = cinput("Test? (y/n)", str, ynValidator) == 'y'
    finder.accuracy(imgs, ovw, test, True)
    # finder.speedBenchmark(imgs, 50, (720, 1280))
    # finder.speedBenchmark(imgs, 50, (480, 640))
    
    
    ib2 = ImageBrowserBehavior(imgs, finder)
    
    while True:
        # ib.loop()
        ib2.loop()