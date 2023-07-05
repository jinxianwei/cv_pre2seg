import numpy as np
import cv2
from typing import Optional

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
            # TODO the png of channel is 4, , cv2.IMREAD_UNCHANGED分情况是否有必要, tiff的数值范围会有影响
            # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = cv2.imread(img_path)
            
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





        return results
