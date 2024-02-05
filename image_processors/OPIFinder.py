from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import cv2
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from cinput import cinput

from image_processors.ImageProcessor import ImageProcessor
from utils import debugScore


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
        def yielding(img:cv2.Mat, choices:Union[Dict[str,str],Callable[[Dict[str,str]], Dict[str,str]]]=None, steps:int=None):
            p = tuple([None])
            args = tuple([img.copy()])
            self.updateChoices()
            choices = self.getChoices(choices)
            if steps is None:
                steps = len(choices)
            for k, v in choices.items():
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
        
        res = yielding(img, choices, steps)
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