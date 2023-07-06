import cv2
from typing import Optional, Dict
from matplotlib import pyplot as plt
import numpy as np

def show(name: str, img: np.ndarray) -> None:
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class Classic_ThresholdAlgorithm():
    """ classic threshold algorithm of segmentation
    """
    def __init__(self, 
                 img_info: Dict) -> Optional[Dict]:
        self.img_info = img_info

    def adaptiveThreshold(self):
        img = self.img_info['img']
        
        th1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        show('adaptiveThreshold', th1)
        show('adaptiveThreshold', th2)
        self.img_info['adaptiveThreshold'] = th2
        return th2
        # raise NotImplementedError
    
    def emThreshold():

        raise NotImplementedError
    
    def rangeThreshold(self,
                       minimun_value: int,
                       maximum_value: int):
        img = self.img_info['img']
        
        ret, th1 = cv2.threshold(img, minimun_value, maximum_value, cv2.THRESH_BINARY)  

        show('rangeThreshold', img)
        self.img_info['rangeThreshold'] = th1
        return th1
        # raise NotImplementedError
    
    def basicThreshold(self,
                       ):

        raise NotImplementedError
    

class Classic_BaseAlgorithm():
    """
    class base 
    """
    def __init__(self, 
                 img_info: Dict) -> Optional[Dict]:
        pass

    def base_Invert(self, ):
        raise NotImplementedError
    
    def base_Blank(self, ):
        raise NotImplementedError
    

class Classic_Edges():
    def __init__(self,
                 img_info: Dict) -> Optional[Dict]:
        pass

    def Watershed(self, ):
        raise NotImplementedError
    
    def Find_Edges(self):
        
        raise NotImplementedError
    
    def Find_Circles(self, ):

        raise NotImplementedError
    
    def Find_Lines(self,):

        raise NotImplementedError
    
    def Advanced_FindText(self, ):
        
        raise NotImplementedError
    
    def Advanced_FindFacialFeatures(self, ):

        raise NotImplementedError
    
class Classic_SNAP():
    def __init__(self, 
                 img_info: Dict) -> Optional[Dict]:
        pass

    def SNAP_AutoSegmentation(self,):

        raise NotImplementedError
    


class Classic_Extrema():
    def __init__(self,
                 img_infol: Dict) -> Optional[Dict]:
        pass
    def Extrema_globalMaximum(self,):

        raise NotImplementedError
    
    def Extrema_globalMinimum(self, ):

        raise NotImplementedError
    
    def Extrema_LocalMaxima(self, ):

        raise NotImplementedError
    
    def Extrema_LocalMinima(self, ):

        raise NotImplementedError
    
class Classic_MLmodel:
    def __init__(self, 
                 img_info: Dict) -> Optional[Dict]:
        
        raise NotImplementedError

    
    