# !usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
 
#color_path：彩色图片路径
#gray_path：灰度图片路径
#dst_color_path：反色后的彩色图片路径
#dst_gray_path：反色后的灰度图片路径
def img_invert(color_path, gray_path, dst_color_path, dst_gray_path):
    #打开彩色原始图像
    img_color = cv2.imread(color_path) 
    #获取彩色图像宽、高、通道数
    height_color, width_color, channels_color = img_color.shape
    #打印彩色图像的宽、高、通道数
    print("color:width[%d],height[%d],channels[%d]" % (width_color, height_color, channels_color))
 
    #将彩色图像转换成灰度图像
    #注意参数2：OpenCV中彩色色彩空间是BGR，转换成灰度色彩空间GRAY
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    #获取灰度图像宽、高、通道数
    #height_gray, width_gray = img_gray.shape
    height_gray, width_gray = img_gray.shape[:2]
    #打印彩色图像的宽、高、通道数
    print("gray:width[%d],height[%d]" % (width_gray, height_gray))
    #保存灰度图片
    cv2.imwrite(gray_path, img_gray)
 
    #显示转换之前的彩色图像和灰度图像
    cv2.imshow('img_color', img_color)
    cv2.imshow('img_gray', img_gray)
 
    #先将彩色图像反色
    #创建空白数组
    for row in range(height_color):
        for col in range(width_color):
            for chan in range(channels_color):
                point = img_color[row, col, chan]
                img_color[row, col, chan] = 255 - point
 
    #显示和保存反色后彩色图像
    cv2.imshow('dst_color', img_color)
    cv2.imwrite(dst_color_path, img_color)
 
    #接着讲灰度图像反色
    #创建空白数组
    for row in range(height_gray):
        for col in range(width_gray):
            img_gray[row][col] = 255 - img_gray[row][col]
 
    #显示和保存反色后灰度图像
    cv2.imshow('dst_gray', img_gray)
    cv2.imwrite(dst_gray_path, img_gray)
 
    #按下任何键盘按键后退出
    cv2.waitKey()
    #销毁所有窗口
    cv2.destroyAllWindows()
    print("test end...")
 
if __name__ == '__main__':
    color_path = 'E:\\cv_pre2seg\\source\\2.jpg'
    gray_path = 'gray.jpg'
    dst_color_path = 'dst_color.jpg'
    dst_gray_path = 'dst_gray.jpg'
    img_invert(color_path, gray_path, dst_color_path, dst_gray_path)