import numpy as np
import cv2
from PIL import Image
from typing import Optional, Dict
import io

MODEL_PALETTE = {
    '1': 1,
    'L': 8,
    'P': 8,
    'RGB': 24,
    'RGBA': 32,
    'CMYK': 32, 
    'YCbCr': 24,
    'I': 32,
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

        try:
            img_path = self.img_path

            

            img_pil = Image.open(img_path)
            temp = img_pil.getbands()
            model = ''
            for i in range(len(temp)):
                model += temp[i]
            # TODO 异常
            bitdepth = MODEL_PALETTE[model]
            results['BitDepth'] = bitdepth



            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            # cv2.imshow('img', img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e

        if self.to_float32:
            img = img.astype(np.float32)

        # results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['height'] = img.shape[0]
        results['width'] = img.shape[1]
        # TODO the channel depend on the flag and cv2.imread args
        # results['channel'] = img.shape[2]
        results['size'] = img.size
        results['dtype'] = img.dtype
        results['ndim'] = img.ndim
        results['nbytes'] = img.nbytes

        results['element_size'] = img.size
        results['itemsize'] = img.itemsize

        

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
