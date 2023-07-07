import cv2
from typing import Optional, Dict
from matplotlib import pyplot as plt
import numpy as np
import copy

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

    def adaptiveThreshold(self) -> np.ndarray:
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
                       maximum_value: int) -> np.ndarray:
        img = self.img_info['img']
        
        ret, th1 = cv2.threshold(img, minimun_value, maximum_value, cv2.THRESH_BINARY)  

        show('rangeThreshold', img)
        self.img_info['rangeThreshold'] = th1
        return th1
        # raise NotImplementedError
    
    def basicThreshold(self, 
                       maximum_value) -> np.ndarray:
        img = self.img_info['img']
        ret, th1 = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
        # 将图片中灰度值超过125的至为255

        show('basicThreshold', th1)
        self.img_info['basicThreshold'] = th1
        return th1
        # raise NotImplementedError
    

class Classic_BaseAlgorithm():
    """
    class base 
    """
    def __init__(self, 
                 img_info: Dict) -> Optional[Dict]:
        self.img_info = img_info

    def base_Invert(self) -> np.ndarray:
        """
        invert the image of color and gray
        """
        img_path = self.img_info['path']
        img_color = cv2.imread(img_path) 
        height_color, width_color, channels_color = img_color.shape
        #将彩色图像转换成灰度图像
        #注意参数2：OpenCV中彩色色彩空间是BGR，转换成灰度色彩空间GRAY
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        height_gray, width_gray = img_gray.shape[:2]
        print(img_gray.shape)

        #显示转换之前的彩色图像和灰度图像
        cv2.imshow('img_color', img_color)
        cv2.imshow('img_gray', img_gray)

        # for row in range(height_color):
        #     for col in range(width_color):
        #         for chan in range(channels_color):
        #             point = img_color[row, col, chan]
        #             img_color[row, col, chan] = 255 - point

        # for row in range(height_gray):
        #     for col in range(width_gray):
        #         img_gray[row][col] = 255 - img_gray[row][col]

        # Note the api of cv2.bitwise_not is more efficient than for loop
        invert_color = cv2.bitwise_not(img_color)
        invert_gray = cv2.bitwise_not(img_gray)

        self.img_info['invert_color'] = invert_color
        self.img_info['invert_gray'] = invert_gray


        cv2.imshow('invert_color', invert_color)
        cv2.imshow('invert_gray', invert_gray)

        #按下任何键盘按键后退出
        cv2.waitKey()
        #销毁所有窗口
        cv2.destroyAllWindows()

        return invert_gray

    def base_Blank(self) -> np.ndarray:
        """
        Creates a blank segmentation. Useful for manually creating seed with
        "Manual Edit" for use with steps such as "Regin Grow", "Active Contour",
        "Local Threshold"
        """
        height = self.img_info['height']
        width = self.img_info['width']
        blank = np.ones([height, width])
        blank *= 255

        self.img_info['blank'] = blank

        cv2.imshow("blank", blank)
        cv2.waitKey()
        cv2.destroyAllWindows()

        return blank
    

class Classic_Edges():
    def __init__(self,
                 img_info: Dict) -> Optional[Dict]:
        self.img_info = img_info

    def Watershed(self) -> np.array:
        img_path = self.img_info['path']
        img = cv2.imread(img_path)
        # Limiarizacao
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        cv2.imshow('Apos limizarizacao',thresh)


        # Remocao de ruidos
        kernel = np.ones((3,3),np.uint8)
        # kernel = np.ones((1,1),np.uint8)
        # opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
        opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 1)

        # Extracao de background
        sure_bg = cv2.dilate(opening,kernel,iterations=3)

        # Extracao de foreground
        # dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
        ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

        # Encontrando regiao desconhecida
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)

        # Desenhando marcadores
        ret, markers = cv2.connectedComponents(sure_fg)

        # Adicionando 1 para todos os rotulos, diferenciando o background, que tera valor 1
        markers = markers+1

        # Marcando a regiao desconhecida com o valor 0
        markers[unknown==255] = 0


        # Aplicando watershed
        markers = cv2.watershed(img,markers)
        img[markers == -1] = [255,0,0]


        show('watershed', img)
        self.img_info['watershed'] = img
        return img

        # raise NotImplementedError
    
    def Find_Edges(self, edge_type: str):
        assert edge_type in {'Sobel', 'Canny', 'Laplacian'}, 'the type of {} is not sopport.'.format(edge_type)
        path = self.img_info['path']
        img = cv2.imread(path)
        if img is None:
            print("Failed to read the image")
        if edge_type == 'Sobel':
            img1 = cv2.Sobel(img, cv2.CV_64F, 1, 0) # x方向边缘检测
            sobelx = cv2.convertScaleAbs(img1)
            img1 = cv2.Sobel(img, cv2.CV_64F, 0, 1)  # y方向边缘检测
            sobely = cv2.convertScaleAbs(img1)
            sobelxy1 = cv2.bitwise_and(sobelx, sobely) # 对两方向上的检测进行与运算
            # sobelxy1 = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)

            cv2.imshow('sobel edges', sobelxy1)
            cv2.waitKey()
            cv2.destroyAllWindows()

            self.img_info['sobel_edge'] = sobelxy1

            return sobelxy1
        elif edge_type == 'Canny':
            img = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(image=gray, threshold1=100, threshold2=200)

            cv2.imshow('canny edges', edges)
            cv2.waitKey()
            cv2.destroyAllWindows()

            self.img_info['canny_edge'] = edges

            return edges    
        
        elif edge_type == 'Laplacian':
            img = cv2.imread(path)
            
            lap = cv2.Laplacian(img, cv2.CV_64F)
            img1 = cv2.convertScaleAbs(lap)

            cv2.imshow('Laplacian edges', img1)
            cv2.waitKey()
            cv2.destroyAllWindows()

            self.img_info['Laplacian_edges'] = img1

            return img1
    
    def Find_Circles(self, ):
        img_path = self.img_info['path']
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像
        # hough transform  规定检测的圆的最大最小半径，不能盲目的检测，否则浪费时间空间
        #                          灰度图像，只支持霍夫圆检测，1为原始大小2为原始大小的一半；圆心之间的最小距离；canny检测器的高阈值
        #                          检测阶段的累加器阈值，越小，会增加不存在的圆，越大检测到的圆形越完美；检测的最小圆半径；最大圆半径
        circle1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=30, minRadius=20, maxRadius=40)

        try:
            circles = circle1[0, :, :]  # 提取为二维, 如果没有提取到圆，直接索引会报错
            circles = np.uint16(np.around(circles))  # 四舍五入，取整
            for i in circles[:]:
                cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 1)  # 画圆
                # cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 3)  # 画圆心
            self.img_info['houghcircle'] = img

            cv2.imshow('houghcircle', img)
            cv2.waitKey()
            cv2.destroyAllWindows()
            return img
        except Exception as e:
            if 'houghcircle' not in self.img_info:
                return None
            else:
                raise e
    
    def Find_Lines(self,):
        path = self.img_info['path']
        img = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,50,150,apertureSize = 3)
        lines = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength=10,maxLineGap=10)


        if lines is None or len(lines) == 0:
            print('this img can not find lines')
        else:
            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
            
            cv2.imshow('lines', img)
            cv2.waitKey()
            cv2.destroyAllWindows()        

        return img
        
    
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
                 img_info: Dict) -> Optional[Dict]:
        self.img_info = img_info

    def Extrema_globalMaximum(self) -> np.ndarray:
        # img = self.img_info['img']
        img_path = self.img_info['path']
        img = cv2.imread(img_path, 0)
        min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(img)
        # min_val
        # 满足条件的保留，不满足条件的设为255-max_val
        max_mask = np.where(img==max_val, img, 255-max_val)

        cv2.imshow('max_mask', max_mask)
        cv2.imshow('ori_img', img)

        cv2.waitKey()
        cv2.destroyAllWindows()
        self.img_info['max_mask'] = max_mask

        return max_mask

    def Extrema_globalMinimum(self, ):
        img_path = self.img_info['path']
        img = cv2.imread(img_path, 0)
        min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(img)
        min_mask = np.where(img==min_val, img, 255-min_val)
 
        cv2.imshow('min_mask', min_mask)
        cv2.imshow('ori_img', img)

        cv2.waitKey()
        cv2.destroyAllWindows()
        self.img_info['min_mask'] = min_mask   

        return min_mask  
    
    def Extrema_LocalMaxima(self,
                            max_val: int):
        img_path = self.img_info['path']
        img = cv2.imread(img_path, 0)
        # min_val, max_val, min_indx, max_indx = cv2.minMaxLoc(img)
        # min_val
        # 满足条件的保留，不满足条件的设为255-max_val
        max_mask = np.where(img<=max_val, img, 255)

        cv2.imshow('max_mask', max_mask)
        cv2.imshow('ori_img', img)

        cv2.waitKey()
        cv2.destroyAllWindows()
        self.img_info['max_mask'] = max_mask

        return max_mask
        
    def Extrema_LocalMinima(self, ):

        raise NotImplementedError
    
class Classic_MLmodel:
    def __init__(self, 
                 img_info: Dict) -> Optional[Dict]:
        
        raise NotImplementedError

    
    