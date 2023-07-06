import numpy as np
import cv2
from PIL import Image
from typing import Optional, Dict
import io
import tifffile

def show(name: str, img: np.ndarray) -> None:
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

MODEL_PALETTE = {
    '1': 1,
    'L': 8,
    'P': 8,
    'RGB': 24,
    'RGBA': 32,
    'CMYK': 32, 
    'YCbCr': 24,
    'I;16': 16,
    'I;32':32,
    'F': 32
}

class LoadImageFromFile():
    """Load an image from file.

    Required Keys:

    - img_path

    Modified Keys:

    - img
    - img_shape

    Args:
        img_path (str): The path of img.
    """

    def __init__(self,
                img_path: str) -> None:
        self.img_path = img_path
        self.ignore_empty = False
        self.to_float32 = False


    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict of img properities.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        img_path = self.img_path
        try:
            results = self.get_filesize(img_path, results)
            results = self.get_bitdepth(img_path, results)
            results = self.get_imginfor(img_path, results)

            if results['tail'] == 'tif':
                img_tif = tifffile.imread(img_path)
                img_tif


        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        return results


    def get_filesize(self, 
                     img_path: str,
                     results: Dict)-> Optional[dict]:
        """
        get img file size
        """
        with open(img_path, 'rb') as f:
            img_buff = f.read()
        img_b = io.BytesIO(img_buff).read()
        bit_size = len(img_b)
        results['FileSize'] = bit_size

        return results
    
    def get_bitdepth(self, 
                     img_path: str,
                     results: dict) -> Optional[dict]:
        """
        get img bitdepth

        Args:

        """
        img_pil = Image.open(img_path)
        # temp = img_pil.getbands()
        # model = ''
        # for i in range(len(temp)):
        #     model += temp[i]
        # # TODO 异常
        # bitdepth = MODEL_PALETTE[model]
        # results['BitDepth'] = bitdepth

        img_info = img_pil.info
        if 'gamma' in img_info:
            results['Gamma'] = img_info['gamma']

        if 'dpi' in img_info:
            results['dpi'] = img_info['dpi']


        mode = img_pil.mode
        results['BitDepth'] = MODEL_PALETTE[mode]
        results['mode'] = img_pil.mode

        

        return results
    
    def get_imginfor(self, 
                     img_path: str,
                     results: dict) -> Optional[dict]:
        """
        get img infor such as h, w, c, and ndarray data

        Args:

        """
        img = cv2.imread(img_path, 0) # 从图像的效果上看，没有太大差异
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_size'] = img.shape[:2]
        results['height'] = img.shape[0]
        results['width'] = img.shape[1]
        # TODO the channel depend on the flag and cv2.imread args
        # if len(img.shape)>2:
        #     results['channel'] = img.shape[2]
        # results['size'] = img.size
        # results['dtype'] = img.dtype
        # results['ndim'] = img.ndim
        # results['nbytes'] = img.nbytes

        # results['element_size'] = img.size
        # results['itemsize'] = img.itemsize

        show('ori', img)
        results['tail'] = img_path.strip().split('.')[-1]

        results['MaxSampleValue'] = img.max()
        results['MinSampleValue'] = img.min()
        return results
