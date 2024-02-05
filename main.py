from cProfile import label
from itertools import chain, permutations, product
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
from colorama import Fore
import cv2
import numpy as np
import imutils
import pytesseract
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

from cinput import cinput, ynValidator

SAMPLES_PATH = Path("samples")
CONVERTED_PATH = Path("converted")

PATHS = [SAMPLES_PATH, CONVERTED_PATH]

KEY_LEFT = 81
KEY_UP = 82
KEY_RIGHT = 83
KEY_DOWN = 84
KEY_ESC = 27

ORANGE = np.array([54, 114, 238], np.int16)

def ensureExists():
    for p in PATHS:
        p.mkdir(parents=True, exist_ok=True)

def convert():
    for p in SAMPLES_PATH.iterdir():
        newPath = CONVERTED_PATH / (p.stem + '.png')
        if newPath.exists():
            continue
        im = cv2.imread(str(p))
        if im is None:
            continue
        cv2.imwrite(str(newPath), im)

class ImageData:
    def __init__(self):
        self.thresh = 100
        self.othresh = 60
        self.lastKey = -1
        self.minArea = 0.008
        self.maxArea = 0.85
        self.erode = 0.001
        self.maxMergeDistance = 20 / 1080
        self.grow = 1.03
        self.margin = 0.001
        self.ocrMinArea = 20 / 1080
        self.ocrMaxArea = 0.15
        self.lineWidth = 0.035
        self.lineBorder = 0.04
        

class ImageBrowser:
    def __init__(self, imgs: List[cv2.Mat]):
        self.imgs = imgs
        self.data = list(ImageData() for _ in imgs)
        self.index = 0
        self.processingMethods:List[Callable[[cv2.Mat, ImageData], cv2.Mat]] = []
        self.methodIndex = 0
        self.colors = []
        self.img = imgs[0]
        
        cv2.namedWindow("main")
        cv2.namedWindow("disp")
        cv2.setMouseCallback('disp', self.mouseEvent)
        cv2.imshow('disp', self.img)
        
        
        
    def mouseEvent(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.colors.append(self.img[y,x])
            arr = np.array(self.colors)
            print(arr.mean(0))

    def loop(self):
        k = cv2.waitKey(10)
        if k == -1:
            return
        
        if k == KEY_LEFT:
            self.index = len(self.imgs) - 1 if self.index == 0 else self.index - 1
        if k == KEY_RIGHT:
            self.index = (self.index + 1) % len(self.imgs)
        if k == KEY_DOWN:
            self.methodIndex = len(self.processingMethods) - 1 if self.methodIndex == 0 else self.methodIndex - 1
        if k == KEY_UP:
            self.methodIndex = (self.methodIndex + 1) % len(self.processingMethods)
        if k == KEY_ESC:
            exit(0)
        
        data = self.data[self.index]
        data.lastKey = k
        method = self.processingMethods[self.methodIndex]
        img = self.imgs[self.index]
        self.img = method(img, data)
        cv2.imshow('disp', self.img)
        cv2.imshow('main', img)
        

    
    
def noProcess(img:cv2.Mat, data:ImageData):
    return img

def normalize(img: cv2.Mat, data: ImageData):
    mn = img.min((0,1))
    mx = img.max((0,1))
    dt = mx-mn
    fimg = img.astype(np.float32)
    return ((fimg[:,:]-mn)/dt*255).astype(np.uint8)

def gray(img: cv2.Mat, data: ImageData, preview=True):
    v = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = normalize(v, data)
    
    if preview:
        for i, col in enumerate(['blue','green','red']):
            cv2.imshow(col, normalize(img[:,:,i], data))
    return res

def red(img: cv2.Mat, data: ImageData, preview=True):
    r = normalize(img[:,:,2], data)
    return r

def redAdaptiveThresh(img: cv2.Mat, data: ImageData, preview=True):
    r:cv2.Mat = red(img, data, False)
    
    
    return r
    

def thresh(img: cv2.Mat, data: ImageData, preview=True):
    v = normalize(img, data)
    r = cv2.threshold(v, data.thresh, 255, cv2.THRESH_BINARY)[1]
    if preview:
        return cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    return r

def orangeness1(img: cv2.Mat, data: ImageData, preview=True):
    v = normalize(img, data)
    # v = img
    dist = np.abs((v.astype(np.int16) - ORANGE))
    nn = dist.mean(2).astype(np.uint16)
    if preview:
        n = normalize(nn, data)
        return cv2.cvtColor(n, cv2.COLOR_GRAY2BGR)
    return nn

def orangeness1Thresh(img: cv2.Mat, data: ImageData, preview=True):
    v = orangeness1(img, data, False).astype(np.uint8)
    if preview:
        if data.lastKey == ord('w'):
            data.othresh += 1
        if data.lastKey == ord('s'):
            data.othresh -= 1
        print(f'othresh = {data.othresh}')
    r = cv2.threshold(v, data.othresh, 255, cv2.THRESH_BINARY)[1]
    if preview:
        return cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
    return r

def orangeness2(img: cv2.Mat, data: ImageData, preview=True):
    # v = normalize(img, data)
    v = img
    
    hsv = cv2.cvtColor(v, cv2.COLOR_BGR2HSV)
    orange = cv2.cvtColor(ORANGE[np.newaxis, np.newaxis].astype(np.uint8), cv2.COLOR_BGR2HSV)
    hue = hsv[:,:,0].astype(np.int16)
    cv2.imshow('test1', hsv[:,:,0])
    cv2.imshow('test2', hsv[:,:,1])
    cv2.imshow('test3', hsv[:,:,2])
    ohue = orange[0,0,0].astype(np.int16)
    orangeness = (np.abs(hue-ohue)*2)
    cv2.imshow('test4', ((orangeness * hsv[:,:,1])/2).astype(np.uint8))
    orangeness[orangeness == 256] -= 1
    orangeness.astype(np.uint8)
    return orangeness

def filterThresh(img: cv2.Mat, data: ImageData, preview=True):
    if preview:
        if data.lastKey == ord('w'):
            data.grow += 0.005
        if data.lastKey == ord('s'):
            data.grow -= 0.005
    # Get orange sections
    v = orangeness1Thresh(img, data, False)
    h,w = v.shape[:2]
    
    # Find the contours
    cnts = cv2.findContours(v.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Get absolute min and max area
    minArea = data.minArea * h*w
    maxArea = data.maxArea * h*w
    
    margin = data.margin * h
    
    mask = np.zeros((h,w), np.uint8)
    candidates = []
    res = img.copy()
    
    def filterContour(contour:np.ndarray):
        # Filter by area
        a = cv2.contourArea(contour)
        if a <= minArea:
            return False, None, None
        if a > maxArea:
            return False, None, None
        # Get angled rect
        r = cv2.minAreaRect(contour)
        (xx, yy), (ww, hh), rr = r
        
        if ww*hh <= minArea:
            return False, None, None
        if ww*hh > maxArea:
            return False, None, None
        box = cv2.boxPoints(r)
        
        if np.any(np.logical_or(box <= (margin, margin), box >= (w-margin, h-margin))):
            return False, None, None
        
        box = np.int_(box)
        return True, box, r
    
    for c in cnts:
        # Filter contours
        select, box, r = filterContour(c)
        
        # Skip rejected
        if not select:
            continue
        
        # Get rect components
        (xx,yy),(ww,hh),rr = r
        
        # Add the candidate to the list
        candidates.append(c)
        
        # Draw some shapes
        if preview:
            res = cv2.drawContours(res, [c], -1, (0,255,0), 1)
            res = cv2.drawContours(res, [box], 0, (255,0,0),1)
        
        # Expand the boxes to allow 2 seperated shapes to become one
        expanded = cv2.boxPoints(((xx,yy),(ww*data.grow, hh*data.grow),rr))
        expanded = np.int_(expanded)
        mask = cv2.drawContours(mask, [expanded], 0, (255),-1)
    
    # Grab the contours of the expanded mask
    mcnts = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mcnts = imutils.grab_contours(mcnts)
    
    # Convert candidates into np array
    candidates = np.array([vvv for vvv in chain(*[values for values in [vv for vv in candidates]])]).squeeze()
    trapezoids = []
    for c in mcnts:
        # Filter contours
        select, box, r = filterContour(c)
        
        # Skip rejected
        if not select:
            continue
        # Get rect components
        (xx,yy),(ww,hh),rr = r
        
        # Draw the rough outline
        if preview:
            res = cv2.drawContours(res, [box], 0, (0,0,255),2)
        
        # Find the trapezoid contained in this expanded shape
        trapezoid = np.zeros((4,2), np.int_)
        
        # Get the clossest contour point for each corner
        for i, corner in enumerate(box):
            # Get all distances
            m = np.linalg.norm(candidates[:] - corner, axis=1, keepdims=True)
            # Find minimum
            idx = np.where(m == m.min())
            # Grab minimum's contour coordinates
            if idx[0].size > 1:
                trapezoid[i] = candidates[idx[0][0]]
            else:
                trapezoid[i] = candidates[idx[0]]
        
        # Draw the resulting trapezoid
        if preview:
            res = cv2.drawContours(res, [trapezoid], 0, (255,255,0),3)
        trapezoids.append(trapezoid)
    
    if preview:
        return res
    return res, trapezoids
    
def straighten(img: cv2.Mat, data: ImageData, preview=True):
    prev, trapezoids = filterThresh(img, data, preview=False)
    if preview:
        cv2.imshow('test', prev)
    
    warps = []
    for i, trapezoid in enumerate(trapezoids):
        # now that we have our rectangle of points, let's compute
        # the width of our new image
        (tl, tr, br, bl) = trapezoid
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        # ...and now for the height of our new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        # take the maximum of the width and height values to reach
        # our final dimensions
        maxWidth = max(int(widthA), int(widthB))
        maxHeight = max(int(heightA), int(heightB))
        # construct our destination points which will be used to
        # map the screen to a top-down, "birds eye" view
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype = "float32")
        # calculate the perspective transform matrix and warp
        # the perspective to grab the screen
        M = cv2.getPerspectiveTransform(np.float32(trapezoid), dst)
        warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
        if preview:
            cv2.imshow(f'test{i}', warp)
        warps.append(warp)
    
    if preview:
        return warp
    return warps, trapezoids
    
def ocr(img: cv2.Mat, data: ImageData, preview=True):
    warps, trapezoids = straighten(img,data,preview=False)
    hh,ww,_ = img.shape
    
    # Get absolute variables
    minArea = hh*ww*data.ocrMinArea
    maxArea = hh*ww*data.ocrMaxArea
    lineWidth = int(hh*data.lineWidth)
    lineBorder = int(hh*data.lineBorder)
    if data.lastKey != -1:
        print(f"Min ocr: {data.ocrMinArea*100}% ({minArea}) | Max ocr: {data.ocrMaxArea*100}% ({maxArea})\nWidth: {data.lineWidth*100}% ({lineWidth}) | Border: {data.lineBorder*100}% ({lineBorder})")
    
    # List for valid results
    results = []
    
    # Iterate over the warped (straightned) images
    for i, warp in enumerate(warps):
        # Get the red channel normalized
        contrast = red(normalize(warp,data), data, preview=False)
        if preview:
            cv2.imshow('test', contrast)
        
        # Iterate over orientations
        for orientation in range(4):
            # Rotate the image
            rotated = np.rot90(contrast, orientation)
            h,w = rotated.shape
            
            # Split top and bottom half
            split = [rotated[:h//2],rotated[h//2:]]
            
            
            score = 0
            
            # Update line values to warped resolution
            lineWidth = max(int(h*data.lineWidth), 2)
            lineBorder = max(int(h*data.lineBorder), 2)
            
            # Prepare middle line checks
            half = h//2
            lower = half-lineWidth//2
            upper = half+lineWidth//2 + 1
            line = rotated[lower:upper]
            border1 = rotated[lower-lineBorder:lower]
            border2 = rotated[upper:upper+lineBorder]
            lm = line.mean(dtype=np.float32)
            bm1 = border1.mean(dtype=np.float32)
            bm2 = border2.mean(dtype=np.float32)
            bm = (bm1+bm2)/2
            
            if preview:
                dbg = cv2.drawContours(cv2.cvtColor(rotated, cv2.COLOR_GRAY2BGR), [np.array([[0,lower-lineBorder],[w, lower-lineBorder],[w,lower],[0,lower]])], 0, (0,255,0), 1)
                dbg = cv2.drawContours(dbg, [np.array([[0,upper+lineBorder],[w, upper+lineBorder],[w,upper],[0,upper]])], 0, (0,255,0), 1)
                dbg = cv2.drawContours(dbg, [np.array([[0,lower],[w, lower],[w,upper],[0,upper]])], 0, (255,0,0), 1)
                cv2.imshow(f'test {i}', dbg)
            
            # If the line and the area around the line are close, decrease score. Otherwise, use the difference as a base score
            if not math.isclose(bm,lm, rel_tol=0.15, abs_tol=4):
                score += bm-lm
            else:
                score -= 30
            
            # Check if the top border is similar to bottom border
            if math.isclose(bm1,bm2, rel_tol=0.05, abs_tol=5):
                score += 10
            else:
                score -= abs(bm1-bm2)*1.5
                
            # Ensure borders are lighter than the line
            if lm < bm1 and lm < bm2:
                score += 20
            else:
                score -= 40
            
            # If score is very low, don't bother with OCR, saves time
            if score < 0:
                continue
            
            # If score is low, verify with OCR
            if score < 50:
                
                found = True
                for ii, s in enumerate(split):
                    txt = pytesseract.image_to_string(s, lang='eng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
                    if txt == '':
                        score -= 20
                        found = False
                        break
                    score += 10

                    # cv2.imshow(f'test{i},{orientation},{ii}', s)
                    print(f"Found [{i},{orientation},{ii}] '{txt}'")
                if found:
                    score += 20
            print(f"score [{i},{orientation}]: {score}")
            if score > 50:
                if not preview:
                    return True, trapezoids[i]
                results.append(rotated)
    if preview:
        for i, result in enumerate(results):
            cv2.imshow(f'result {i}', result)
    else:
        return False, None
    return warp
    
def canny(img: cv2.Mat, data: ImageData, preview=True):
    v = orangeness1Thresh(img, data, False)
    return cv2.Canny(v,0,1)

def final(img: cv2.Mat, data: ImageData, preview=True):
    success, trapezoid = ocr(img, data, preview=False)
    res = img.copy()
    if success:
        res = cv2.drawContours(res, [trapezoid], 0, (255,0,0), 3)
    return res


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
        if self.debug:
            cv2.imshow(self.__class__.__name__, img)
    
    def destroyWindow(self):
        if self.debug:
            cv2.destroyWindow(self.__class__.__name__)
    
    def speedBenchmark(self, iterations:int, *args, **kwargs) -> np.ndarray[np.float_]:
        res = np.zeros((iterations), np.float_)
        for i in range(iterations):
            start = time.time()
            self.__call__(*args, **kwargs)
            end = time.time()
            res[i] = end-start
        return res

class Normalizer(ImageProcessor):
    def __init__(self):
        super().__init__()
        
    def __call__(self, img:cv2.Mat) -> cv2.Mat:
        pass
    
class SimpleNormalizer(Normalizer):
    def __init__(self):
        super().__init__()
        
    def __call__(self, img:cv2.Mat) -> cv2.Mat:
        mn = img.min((0,1))
        mx = img.max((0,1))
        dt = mx-mn
        fimg = img.astype(np.float32)
        res = ((fimg[:,:]-mn)/dt*255).astype(np.uint8)
        self.debugImg(res)
        return res
    
class Contraster(ImageProcessor):
    def __init__(self):
        super().__init__()
        
    def __call__(self, img: cv2.Mat) -> cv2.Mat:    
        pass
    
class MeanRGBOffset(Contraster):
    def __init__(self, color=ORANGE):
        super().__init__()
        self.color = color
        
    def __call__(self, img: cv2.Mat) -> cv2.Mat:
        dist = np.abs(img.astype(np.int16) - ORANGE)
        res = dist.mean(2).astype(np.uint8)
        self.debugImg(res)
        return res

class Red(Contraster):
    def __init__(self, postNormalizer: Normalizer = None):
        super().__init__()
        self.postNormalizer = postNormalizer
        
    def __call__(self, img: cv2.Mat) -> cv2.Mat:
        res = img[:,:,2]
        if self.postNormalizer is not None:
            res = self.postNormalizer(res)
        self.debugImg(res)
        return res
    
class Thresholder(ImageProcessor):
    def __init__(self):
        super().__init__()
    
    def __call__(self, orangeness:cv2.Mat) -> cv2.Mat:
        pass

class SimpleThreshold(Thresholder):
    def __init__(self, threshold:int):
        super().__init__()
        self.threshold = threshold
    
    def __call__(self, orangeness:cv2.Mat) -> cv2.Mat:
        res = cv2.threshold(orangeness, self.threshold, 255, cv2.THRESH_BINARY)[1]
        self.debugImg(res)
        return res

class EdgeDetector(ImageProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, threshold:cv2.Mat) -> Sequence[cv2.Mat]:
        pass    

class SimpleEdgeDetect(EdgeDetector):
    def __init__(self):
        super().__init__()

    def __call__(self, threshold:cv2.Mat) -> Sequence[cv2.Mat]:
        # Find the contours
        cnts = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        return cnts
def expandRect(rect:Tuple[Tuple[float,float],Tuple[float,float],float]) -> Tuple[float,float,float,float,float]:
    (x, y), (w, h), r = rect
    return x,y,w,h,r

def compactRect(rect:Union[np.ndarray[np.float_], Tuple[float,float,float,float,float]]) -> Tuple[Tuple[float,float],Tuple[float,float],float]:
    return (rect[0], rect[1]), (rect[2], rect[3]), rect[4]

class EdgeFilter(ImageProcessor):
    def __init__(self):
        super().__init__()
        
    def __call__(self, img:cv2.Mat, cnts:Sequence[cv2.Mat]) -> Tuple[np.ndarray[np.bool_],np.ndarray[np.float_],np.ndarray[np.float_],np.ndarray[np.float_]]:
        pass
    
    
class SimpleEdgeFilter(EdgeFilter):
    def __init__(self, minArea:float, maxArea:float, margin:float):
        super().__init__()
        self.minArea = minArea
        self.maxArea = maxArea
        self.margin = margin
        
    def __call__(self, img:cv2.Mat, cnts:Sequence[cv2.Mat]) -> Tuple[np.ndarray[np.bool_],np.ndarray[np.float_],np.ndarray[np.float_],np.ndarray[np.float_]]:
        height, width, _ = img.shape
        minArea = self.minArea * height * width
        maxArea = self.maxArea * height * width
        margin = self.margin * height
        
        areas = np.array([cv2.contourArea(contour) for contour in cnts])
        valid = (areas > minArea) & (areas < maxArea)
        if not np.any(valid):
            return valid, None, None, None
        
        rects = np.array([expandRect(cv2.minAreaRect(c)) if v else (0,0,0,0,0)  for v, c in zip(valid, cnts)])
        rectAreas = rects[:,2] * rects[:,3]
        valid &= rectAreas < maxArea
        
        if not np.any(valid):
            return valid, None, None, None
        
        boxes = np.array([cv2.boxPoints(compactRect(r)) if v else np.zeros((4,2),np.float_) for v, r in zip(valid, rects)])
        
        valid &= np.any((boxes > (margin, margin)) & (boxes < (width-margin, height-margin)), (1,2))
        
        return valid, rects, rectAreas, boxes

class ShapePostProcessor(ImageProcessor):
    def __init__(self):
        super().__init__()
    
    def __call__(self, img:cv2.Mat, rects:np.ndarray[np.float_], valid:np.ndarray[np.bool_], cnts:Sequence[cv2.Mat]) -> Tuple[np.ndarray[np.bool_],np.ndarray[np.float_],np.ndarray[np.float_],np.ndarray[np.float_], Sequence[cv2.Mat]]:
        pass

class FillPostProcess(ShapePostProcessor):
    def __init__(self, edgeDetector:EdgeDetector, edgeFilter:EdgeFilter, grow:float):
        super().__init__()
        self.edgeDetector = edgeDetector
        self.edgeFilter = edgeFilter
        self.grow = grow
    
    def __call__(self, img:cv2.Mat, rects:np.ndarray[np.float_], valid:np.ndarray[np.bool_], cnts:Sequence[cv2.Mat]) -> Tuple[np.ndarray[np.bool_],np.ndarray[np.float_],np.ndarray[np.float_],np.ndarray[np.float_], Sequence[cv2.Mat]]:
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
        cnts2 = self.edgeDetector(mask)
        valid2, rects, rectAreas, boxes = self.edgeFilter(img, cnts2)
        return valid, rects, rectAreas, boxes, cnts
        

# Function to find overlapping rotated rectangles
def findOverlappingRotatedRectangles(img: cv2.Mat, rectangles:np.ndarray[np.float_], valid: np.ndarray[np.bool_], grow:float):
    overlapping_pairs = []
    grown = rectangles.copy()
    grown[:,2:4] *= grow
    oldValid = valid.copy()
    valid[:] = False
    for i, rect1, grown1 in zip(range(len(rectangles)), rectangles, grown):
        if not oldValid[i]:
            continue
        for j, rect2, grown2 in zip(range(i+1, len(rectangles)), rectangles[i+1:], grown[i+1:]):
            if not oldValid[j]:
                continue
            inside = cv2.rotatedRectangleIntersection(compactRect(rect1), compactRect(rect2))
            if inside[0] != cv2.INTERSECT_NONE and len(inside[1]) == 4:
                continue
            intersection = cv2.rotatedRectangleIntersection(compactRect(grown1), compactRect(grown2))
            
            if intersection[0] == cv2.INTERSECT_PARTIAL:
                overlapping_pairs.append((grown1, grown2))
                valid[i] = True
                valid[j] = True
                
            
            # dbg = img.copy()
            # if intersection[0] == cv2.INTERSECT_PARTIAL:
            #     overlapping_pairs.append((grown1, grown2))
            #     dbg = cv2.drawContours(dbg, [np.int_(cv2.boxPoints(compactRect(grown1)))], 0, (0,255,0), 2)
            #     dbg = cv2.drawContours(dbg, [np.int_(cv2.boxPoints(compactRect(grown2)))], 0, (0,255,0), 2)
            # else:
            #     dbg = cv2.drawContours(dbg, [np.int_(cv2.boxPoints(compactRect(rect1)))], 0, (0,0,255), 2)
            #     dbg = cv2.drawContours(dbg, [np.int_(cv2.boxPoints(compactRect(rect2)))], 0, (0,0,255), 2)
            #     dbg = cv2.drawContours(dbg, [np.int_(cv2.boxPoints(compactRect(grown1)))], 0, (255,0,0), 1)
            #     dbg = cv2.drawContours(dbg, [np.int_(cv2.boxPoints(compactRect(grown2)))], 0, (255,0,0), 1)
            # cv2.imshow('overlap', dbg)
            # while cv2.waitKey(100) == -1:
            #     pass

    return np.array(overlapping_pairs)

# Function to compute the minimum area rectangle for a set of rotated rectangles
def minAreaRectRotatedRects(rects: np.ndarray[np.float_]):
    points = []
    for rect in rects:
        box = cv2.boxPoints(compactRect(rect))
        points.extend(box)
    return expandRect(cv2.minAreaRect(np.array(points)))
        
class RectMergePostProcess(ShapePostProcessor):
    def __init__(self, edgeFilter:EdgeFilter, grow:float):
        super().__init__()
        self.edgeFilter = edgeFilter
        self.grow = grow
    
    def __call__(self, img:cv2.Mat, rects:np.ndarray[np.float_], valid:np.ndarray[np.bool_], cnts:Sequence[cv2.Mat]) -> Tuple[np.ndarray[np.bool_],np.ndarray[np.float_],np.ndarray[np.float_],np.ndarray[np.float_],Sequence[cv2.Mat]]:
        if valid is None or not np.any(valid):
            return valid, None, None, None, cnts
        r = rects.copy()
        
        minRects = np.array([minAreaRectRotatedRects(pair) for pair in findOverlappingRotatedRectangles(img, r, valid, self.grow)], ndmin=2)
        if self.debug and minRects.shape[-1] != 0:
            dbg = img.copy()
            for rr in minRects:
                dbg = cv2.drawContours(dbg, [np.int_(cv2.boxPoints(compactRect(rr)))], 0, (0,255,0), 2)
                
            for rr in r:
                dbg = cv2.drawContours(dbg, [np.int_(cv2.boxPoints(compactRect(rr)))], 0, (255,0,0), 1)
                
            self.debugImg(dbg)
        
        areas = minRects[:,2:4].prod(1)
        if len(minRects) != 1:
            boxes = np.array([np.int_(cv2.boxPoints(compactRect(rr))) for rr in minRects])
        else:
            boxes = np.empty((1,0), np.float_)
        return valid, minRects, areas, boxes, cnts

class TrapezoidFinder(ImageProcessor):
    def __init__(self):
        super().__init__()
        
    def __call__(self, img:cv2.Mat, cnts: Sequence[cv2.Mat], valid: np.ndarray[np.bool_], rects: np.ndarray[np.float_], boxes:np.ndarray[np.float_]) -> np.ndarray[np.float_]:
        pass

class NearestContour(TrapezoidFinder):
    def __init__(self):
        super().__init__()
        
    def __call__(self, img:cv2.Mat, cnts: Sequence[cv2.Mat], valid: np.ndarray[np.bool_], rects: np.ndarray[np.float_], boxes:np.ndarray[np.float_]) -> np.ndarray[np.float_]:
        if valid is None or not np.any(valid):
            return None
        res = None
        
        candidates = np.array([vvv for vvv in chain(*[values for values in [vv[1] for vv in filter(lambda c: c[0], zip(valid, cnts))]])]).squeeze()
        trapezoids = []
        if boxes is None:
            return None
        for b in boxes:
            # Draw the rough outline
            if self.debug:
                res = cv2.drawContours(img.copy(), [np.int_(b)], 0, (0,0,255),2)
            
            # Find the trapezoid contained in this expanded shape
            trapezoid = np.zeros((4,2), np.int_)
            
            # Get the clossest contour point for each corner
            for i, corner in enumerate(b):
                # Get all distances
                m = np.linalg.norm(candidates[:] - corner, axis=1, keepdims=True)
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
    
class TrapezoidRectifier(ImageProcessor):
    def __init__(self):
        super().__init__()
        
    def __call__(self, img: cv2.Mat, trapezoids:np.ndarray[np.float_]) -> List[cv2.Mat]:
        pass

class BasicRectify(TrapezoidRectifier):
    def __init__(self):
        super().__init__()
        
    def __call__(self, img: cv2.Mat, trapezoids:np.ndarray[np.float_]) -> List[cv2.Mat]:
        if trapezoids is None:
            return None
        warps = []
        for i, trapezoid in enumerate(trapezoids):
            # now that we have our rectangle of points, let's compute
            # the width of our new image
            (tl, tr, br, bl) = trapezoid
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            # ...and now for the height of our new image
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            # take the maximum of the width and height values to reach
            # our final dimensions
            maxWidth = max(int(widthA), int(widthB))
            maxHeight = max(int(heightA), int(heightB))
            # construct our destination points which will be used to
            # map the screen to a top-down, "birds eye" view
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype = "float32")
            # calculate the perspective transform matrix and warp
            # the perspective to grab the screen
            M = cv2.getPerspectiveTransform(np.float32(trapezoid), dst)
            warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
            if self.debug:
                cv2.imshow(f'test{i}', warp)
            warps.append(warp)
        
        return warps

def debugScore(score:Dict[str,float], ndigits=2):
    s = ''
    for n, v in score.items():
        no = n.startswith('no')
        v = round(v, ndigits)
        if not no:
            s += f'\n{n}: '
        else:
            s += ' | '
        
        color = Fore.GREEN if no != (v > 0.5) else Fore.RED
        s += f'{color}{v}{Fore.RESET}'
    print(s)

class ShapeIdentifier(ImageProcessor):
    def __init__(self):
        super().__init__()
        
    def debugScore(self, score:Dict[str,float], ndigits=2):
        if not self.debug:
            return
        debugScore(score, ndigits)
        
    def __call__(self, img: cv2.Mat, warps: List[cv2.Mat]) -> Tuple[List[Dict[str, float]], List[cv2.Mat]]:
        pass
    
class BasicShapeIdentifier(ShapeIdentifier):
    def __init__(self, lineWidth:float, lineBorder:float, contraster: Contraster):
        super().__init__()
        self.lineWidth = lineWidth
        self.lineBorder = lineBorder
        self.contraster = contraster
        
    def __call__(self, img: cv2.Mat, warps: List[cv2.Mat], rectAreas:np.ndarray[np.float_]) -> Tuple[List[Dict[str, float]], List[cv2.Mat]]:
        if warps is None:
            return None, None
        # List for valid results
        results = []
        scores = []
        
        # Iterate over the warped (straightned) images
        for i, warp in enumerate(warps):
            # Get the red channel normalized
            contrast = self.contraster(warp)
            
            
            # Iterate over orientations
            for orientation in range(4):
                # Rotate the image
                rotated = np.rot90(contrast, orientation)
                h,w = rotated.shape
                
                # Split top and bottom half
                split = [rotated[:h//2],rotated[h//2:]]
                
                
                score = {
                    'lineBorder' : 0.0,
                    'noLineBorder' : 0.0,
                    'topBottomBorders' : 0.0,
                    'noTopBottomBorders' : 0.0,
                    'bordersLighter' : 0.0,
                    'noBordersLighter' : 0.0,
                    'topOCR' : 0.0,
                    'bottomOCR' : 0.0,
                }
                
                # Update line values to warped resolution
                lineWidth = max(int(h*self.lineWidth), 2)
                lineBorder = max(int(h*self.lineBorder), 2)
                
                # Prepare middle line checks
                half = h//2
                lower = half-lineWidth//2
                upper = half+lineWidth//2 + 1
                line = rotated[lower:upper]
                border1 = rotated[lower-lineBorder:lower]
                border2 = rotated[upper:upper+lineBorder]
                lm = line.mean(dtype=np.float32)
                bm1 = border1.mean(dtype=np.float32)
                bm2 = border2.mean(dtype=np.float32)
                bm = (bm1+bm2)/2
                
                if self.debug:
                    dbg = cv2.drawContours(cv2.cvtColor(rotated.copy(), cv2.COLOR_GRAY2BGR), [np.array([[0,lower-lineBorder],[w, lower-lineBorder],[w,lower],[0,lower]])], 0, (0,255,0), 1)
                    dbg = cv2.drawContours(dbg, [np.array([[0,upper+lineBorder],[w, upper+lineBorder],[w,upper],[0,upper]])], 0, (0,255,0), 1)
                    dbg = cv2.drawContours(dbg, [np.array([[0,lower],[w, lower],[w,upper],[0,upper]])], 0, (255,0,0), 1)
                    self.debugImg(dbg)
                
                # If the line and the area around the line are close, decrease score. Otherwise, use the difference as a base score
                if not math.isclose(bm,lm, rel_tol=0.15, abs_tol=4):
                    score['lineBorder'] = (bm-lm)/255
                else:
                    score['noLineBorder'] = 1
                
                # Check if the top border is similar to bottom border
                if math.isclose(bm1,bm2, rel_tol=0.05, abs_tol=5):
                    score['topBottomBorders'] = 1
                else:
                    score['noTopBottomBorders'] = abs(bm1-bm2)/255
                    
                # Ensure borders are lighter than the line
                if lm < bm1 and lm < bm2:
                    score['bordersLighter'] = 1
                else:
                    score['noBordersLighter'] = 1
                
                # # If score is very low, don't bother with OCR, saves time
                # if score < 0:
                #     continue
                
                # If score is low, verify with OCR
                ocrCount = 0
                for ii, s, k in zip(range(2), split, ['topOCR', 'bottomOCR']):
                    txt = pytesseract.image_to_string(s, lang='eng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
                    txt:str
                    txt.rstrip('\n')
                    score[k] = len(txt)
                    if txt == '':
                        continue
                    ocrCount+=1

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
                    # cv2.imshow(f'test{i},{orientation},{ii}', s)
                    print(f"Found [{i},{orientation},{ii}] '{txt}'")
                results.append(rotated)
                scores.append(score)
        if self.debug:
            for s, r in zip(scores, results):
                self.debugScore(s)
                self.debugImg(r)
                while cv2.waitKey(100) == -1:
                    pass
        return scores, results

class ShapeSelector(ImageProcessor):
    def __init__(self):
        super().__init__()
        
    def __call__(self, img: cv2.Mat, scores:List[Dict[str, float]], warps:List[cv2.Mat], trapezoids:np.ndarray[np.float_]) -> Tuple[List[int], List[np.ndarray[np.float_]]]:
        """Classifies shapes

        Args:
            img (cv2.Mat): Input image, used for debugging.
            scores (List[Dict[str, float]]): Scores per trapezoid.
            warps (List[cv2.Mat]): Warps for each trapezoid. Shape is (n, height of n, width of n)
            trapezoids (np.ndarray[np.float_]): Trapezoids on which data is based. Shape is (n, 4, 2)

        Returns:
            Tuple[List[int], List[np.ndarray[np.float_]]]: (indices, selected trapezoids)

        """
        
class LogisticRegressionSelector(ShapeSelector):
    def __call__(self, img: cv2.Mat, scores: List[Dict[str, float]], warps: List[cv2.Mat], trapezoids: np.ndarray[np.float_]) -> Tuple[List[int], List[np.ndarray[np.float_]]]:
        if scores is None:
            return None, None
        # Assuming scores are normalized
        scores_array = np.array([list(score.values()) for score in scores])
        model = LogisticRegression()
        model.fit(scores_array, range(len(scores)))

        # Classify trapezoids
        predictions = model.predict(scores_array)

        selected_indices = list(predictions//4)
        selected_trapezoids = trapezoids[selected_indices]

        return selected_indices, selected_trapezoids

class OPIFinder:
    def __init__(self):
        self.steps = {
            'normalize' : {},
            'orangeness' : {},
            'threshold' : {},
            'edgeDetect' : {},
            'edgeFilter' : {},
            'postProcessShapes' : {},
            'findTrapezoids' : {},
            'rectifyTrapezoids' : {},
            'scoreShapes' : {},
            'selectShapes' : {},
        }
        self.choices = {}
        for k in self.steps.keys():
            self.choices[k] = None
        
        # (passthrough, args)
        self.interStep:Dict[str,Callable[[List[Any], List[Any]], Tuple[List[Any], List[Any]]]] = {
            # input: ([None], [img])
            'normalize' : lambda p, args: (args, args), 
            # output: ([img], [img])
            # call (*[img]) -> (normalized)
            
            # input: ([img], [normalized])
            'orangeness' : lambda p, args: (p, args),
            # output: ([img], [normalized])
            # call (*[normalized]) -> (orangeness)
            
            # input: ([img], [orangeness])
            'threshold' : lambda p, args: (p, args),
            # output: ([img], [orangeness])
            # call (*[orangeness]) -> (threshold)
            
            # input: ([img], [threshold])
            'edgeDetect' : lambda p, args: (p, args),
            # output: ([img], [threshold])
            # call (*[threshold]) -> (cnts)
            
            # input: ([img], [cnts])
            'edgeFilter' : lambda p, args: ((p[0], args), (p[0], args)),
            # output: ([img, cnts], [img, cnts])
            # call (*[img, cnts]) -> (valid, rects, rectAreas, boxes)
            
            # input: ([img, cnts], [valid, rects, rectAreas, boxes])
            'postProcessShapes' : lambda p, args: (p, (p[0], args[1], args[0], p[1])),
            # output: ([img, cnts], [img, valid, rects, rectAreas, boxes])
            # call (*[img, rects, valid, cnts]) -> (valid, rects, rectAreas, boxes, cnts)
            
            # input: ([img, cnts], [valid, rects, rectAreas, boxes, cnts])
            'findTrapezoids' : lambda p, args: ((p[0], args[4], args[2]), (p[0], args[4], args[0], args[1], args[3])),
            # output: ([img, cnts, rectAreas], [img, cnts, valid, rects, boxes])
            # call (*[img, cnts, valid, rects, boxes]) -> (trapezoids)
            
            # input: ([img, cnts, rectAreas], [trapezoids])
            'rectifyTrapezoids' : lambda p, args: ((p[0],p[1],p[2], args[0]), (p[0], args[0])),
            # output: ([img, cnts, rectAreas, trapezoids], [img, trapezoids])
            # call (*[img, trapezoids]) -> warps
            
            # input: ([img, cnts, rectAreas, trapezoids], [warps])
            'scoreShapes' : lambda p, args: (p, (p[0], args[0], p[2])),
            # output: ([img, cnts, rectAreas, trapezoids], [img, warps, rectAreas])
            # call (*[img, warps, rectAreas]) -> (scores, results)
            
            # input: ([img, cnts, rectAreas, trapezoids], [scores, results])
            'selectShapes' : lambda p, args: (p, (p[0], args[0], args[1], p[3])),
        }

    def updateChoices(self):
        for k, v in self.choices.items():
            if v is None:
                try:
                    self.choices[k] = list(self.steps[k].keys())[0]
                except IndexError:
                    pass

    def getChoices(self, choices:Union[Dict[str,str],Callable[[Dict[str,str]], Dict[str,str]]]=None):
        if choices is None:
            return self.choices
        if callable(choices):
            return choices(self.choices.copy())
        

    def getSelectedStep(self, step:str, choices:Union[Dict[str,str],Callable[[Dict[str,str]], Dict[str,str]]]=None):
        choices = self.getChoices(choices)
        s = choices[step]
        if s is None:
            return None
        return self.steps[step][s]
    
    def getStepNumber(self, step:int, choices:Union[Dict[str,str],Callable[[Dict[str,str]], Dict[str,str]]]=None):
        choices = self.getChoices(choices)
        
        return self.getSelectedStep(list(choices.keys())[step])

    def find(self, img:cv2.Mat, choices:Union[Dict[str,str],Callable[[Dict[str,str]], Dict[str,str]]]=None) -> Tuple[bool, Union[np.ndarray[np.int_,np.int_],None]]:
        self.updateChoices()
        # Normalize
        normalized = self.getSelectedStep('normalize', choices)(img)
        # Get orangeness
        orangeness = self.getSelectedStep('orangeness', choices)(normalized)
        # Get threshold
        thresh = self.getSelectedStep('threshold', choices)(orangeness)
        # Edge detect
        cnts = self.getSelectedStep('edgeDetect', choices)(thresh)
        # Filter edges
        valid, rects, rectAreas, boxes = self.getSelectedStep('edgeFilter', choices)(img, cnts)
        if not np.any(valid):
            return False, None
        # Find full shape
        valid, rects, rectAreas, boxes, cnts = self.getSelectedStep('postProcessShapes', choices)(img, rects, valid, cnts)
        if not np.any(valid):
            return False, None
        # Get trapezoids
        trapezoids = self.getSelectedStep('findTrapezoids', choices)(img, cnts, valid, rects, boxes)
        # Rectify trapezoids
        warps = self.getSelectedStep('rectifyTrapezoids', choices)(img, trapezoids)
        for i, w in enumerate(warps):
            cv2.imshow(f'warp {i}', w)
        # Identify shapes
        scores, results = self.getSelectedStep('scoreShapes', choices)(img, warps, rectAreas)
        
        # for s, r in zip(scores, results):
        #     cv2.imshow(f'result: {s}', r)
            
    def find2(self, img:cv2.Mat, yields:bool = False, choices:Union[Dict[str,str],Callable[[Dict[str,str]], Dict[str,str]]]=None, steps:int=None):
        def yielding(img:cv2.Mat, yields:bool = False, choices:Union[Dict[str,str],Callable[[Dict[str,str]], Dict[str,str]]]=None, steps:int=None):
            p = tuple([None])
            args = tuple([img.copy()])
            self.updateChoices()
            choices = self.getChoices(choices)
            if steps is None:
                steps = len(choices)
            for _, k, v in zip(range(steps), choices.keys(), choices.values()):
                if v is None:
                    break
                p, args = self.interStep[k](p, args)
                yield (p, args)
                args = self.steps[k][v](*args)
                
                if not isinstance(args, tuple):
                    args = tuple([args])
                if not isinstance(p, tuple):
                    p = tuple([p])
            yield (p, args)
        
        res = yielding(img, yields, choices, steps)
        if yields:
            return res
        return list(res)[-1]
        
    
    
    def speedBenchmark(self, imgs:Sequence[cv2.Mat], iterations:int, res:Tuple[int, int] = None):
        count = len(imgs)
        if res is not None:
            def resize(imgs: List[cv2.Mat]):
                for img in imgs:
                    h, w, _ = img.shape
                    hh, ww = res
                    ratio = w/h
                    rratio = ww/hh
                    
                    if ratio > 1: # Horizontal image
                        if rratio < 1: # Normalize the ratio to horizontal ex: 16:9
                            rratio = 1/rratio
                    else: # Vertical image
                        if rratio > 1: # Normalize the ratio to vertical ex: 16:9
                            rratio = 1/rratio
                    
                    if ratio > rratio: # Original image has wider ratio than expected result
                        # In this case, we crop the sides
                        hhh = int(round(hh)) # Height will be the same
                        www = int(round(hhh * ratio)) # Get the ratio accurate width
                    else: 
                        # Otherwise, we crop the top and bottom
                        www = int(round(ww)) # Width will be the same
                        hhh = int(round(www / ratio)) # Get the ratio accurate height
                        
                    resized = cv2.resize(img, (hhh, www)) # Resize image
                    deltaLR = (www - ww) // 2
                    deltaUD = (hhh - hh) // 2
                    yield resized[deltaUD:hhh-deltaUD,deltaLR:www-deltaLR] # Crop image
                    
            imgs = resize(imgs)
                    
                    
        datas = list(list(self.find2(img, True)) for img in tqdm(imgs, "Grabbing data", count, unit='img'))
        
        width = 1
        for s in self.steps.items():
            width = max(width, len(s))
        height = len(self.steps)
        for i, stepName, behaviors in tqdm(zip(range(height), self.steps.keys(), self.steps.values()), "Steps", len(self.steps), unit='stp'):
            # splt = plt.subplot(height, 1, i+1, label=stepName)
            splt = plt
            
            labels = []
            x_axis = np.arange(len(datas))
            ww = 1/(len(behaviors) + 1)
            off = lambda id: id*ww
            for ii, bname, b in tqdm(zip(range(len(behaviors)), behaviors.keys(), behaviors.values()), stepName, len(behaviors), unit='bhv'):
                b:ImageProcessor
                labels.append(bname)
                avgs = np.zeros(len(datas), np.float_)
                for iii in tqdm(range(len(datas)), "Benchmarking", len(datas), unit='img'):
                    p, args = datas[iii][i]
                    results = b.speedBenchmark(iterations, *args)
                    avgs[iii] = results.mean()
            
                splt.bar(x_axis + off(ii), avgs, ww, label=bname)
            plt.legend()
            plt.show()
            
    def accuracy(self, imgs: Sequence[cv2.Mat], overwrite:bool, test:bool, validity:bool):
        def save(rScores, rExpected):
            res = np.array([(s, e) for s, e in zip(rScores.values(), rExpected.values())])
            np.save("cache", res)
            
            
        names = [
            'lineBorder',
            'noLineBorder',
            'topBottomBorders',
            'noTopBottomBorders',
            'bordersLighter',
            'noBordersLighter',
            'topOCR',
            'bottomOCR',
            # 'valid',
        ]
        
        def load():
            rScores = {}
            rExpected = {}
            ss = np.load("cache.npy")
            for k, s, e in zip(names, ss[:,0], ss[:,1]):
                rScores[k] = s
                rExpected[k] = e
            return rScores, rExpected
        
        if overwrite:
            
            scores = []
            warps = []
            expected = []
            
            for i, img in enumerate(imgs):
                if test:
                    _, (ss, ww) = self.find2(img)
                if ww is None or ss is None:
                    continue
                if 'valid' not in ss:
                    ss['valid'] = 0
                for s, w in zip(ss, ww):
                    scores.append(s)
                    warps.append(w)
                    exp = {}
                    cv2.imshow('accuracy', w)
                    cv2.waitKey(100)
                    print("\n\n\n")
                    print("Result:")
                    debugScore(s)
                    for k in s.keys():
                        
                        v = cinput(f"Enter expected result for {k}: ", float, 
                                    parser=lambda x: (True, s[k]) if x == '' else (True, float(x)))
                        exp[k] = v
                    
                        
                        
                        
                    # while True:
                    #     kk = cv2.waitKey(10)
                    #     if kk == KEY_ESC:
                    #         exit(0)
                    # for k in s.keys():
                    print("Expected:")
                    debugScore(exp)
                    expected.append(exp)
                
            reorderedScores = {}
            reorderedExpected = {}
            for i, s, e in zip(range(len(scores)), scores, expected):
                for k, ss, ee in zip(s.keys(), s.values(), e.values()):
                    if k not in reorderedScores.keys():
                        reorderedScores[k] = np.zeros((len(scores)), np.float_)
                        reorderedExpected[k] = np.zeros((len(scores)), np.float_)
                    reorderedScores[k][i] = ss
                    reorderedExpected[k][i] = ee
            save(reorderedScores, reorderedExpected)
        else:
            scores = []
            _, reorderedExpected = load()
            if 'valid' not in reorderedExpected:
                reorderedExpected['valid'] = np.zeros((len(reorderedExpected[names[0]])), np.float_)
            for i, img in enumerate(imgs):
                _, (ss, ww) = self.find2(img)
                if ww is None or ss is None:
                    continue
                
                for s, w in zip(ss, ww):
                    scores.append(s)
                    if validity:
                        cv2.imshow('valid', w)
                        cv2.waitKey(100)
                        v = cinput("Is this a valid image?", float)
                        reorderedExpected['valid'][i] = v
            reorderedScores = {}
            for i, s in zip(range(len(scores)), scores):
                for k, ss in zip(names, s.values()):
                    if k not in reorderedScores.keys():
                        reorderedScores[k] = np.zeros((len(scores)), np.float_)
                    reorderedScores[k][i] = ss
        if validity:
            save(reorderedScores, reorderedExpected)
        
        
        for k, s, e in zip(reorderedScores.keys(), reorderedScores.values(), reorderedExpected.values()):
            dbg = f"{k}:\n"
            s = s>=0.5
            e = e>=0.5
            truePos = (e == True) & (s == True)
            falsePos = (e == False) & (s == True)
            trueNeg = (e == False) & (s == False)
            falseNeg = (e == True) & (s == False)
            for name, val in [('true pos', truePos[e==True]), ('false neg', falseNeg[e==True]), ('true neg', trueNeg[e==False]), ('false pos', falsePos[e==False])]:
                val = val.sum()
                ratio = round(val / len(s) * 100, 1)
                dbg += f"{name.ljust(9)}: {str(val).rjust(3)}/{str(len(s)).ljust(3)} {ratio}%\n"
            print(dbg)

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
    finder.steps['normalize']['simple'] = SimpleNormalizer()
    
    finder.steps['orangeness']['meanRGBOffset'] = MeanRGBOffset()
    
    finder.steps['threshold']['simple'] = SimpleThreshold(50)
    
    finder.steps['edgeDetect']['simple'] = SimpleEdgeDetect()
    
    finder.steps['edgeFilter']['simple'] = SimpleEdgeFilter(0.008, 0.9, 0.001)
    
    finder.steps['postProcessShapes']['fillExpand'] = FillPostProcess(finder.steps['edgeDetect']['simple'], finder.steps['edgeFilter']['simple'], 1.1)
    finder.steps['postProcessShapes']['rectMerge'] = RectMergePostProcess(finder.steps['edgeFilter']['simple'], 1.1)
    
    finder.steps['findTrapezoids']['simple'] = NearestContour()
    
    finder.steps['rectifyTrapezoids']['simple'] = BasicRectify()
    
    redContraster = Red(SimpleNormalizer())
    
    finder.steps['scoreShapes']['simple'] = BasicShapeIdentifier(0.035, 0.04, redContraster)
    
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