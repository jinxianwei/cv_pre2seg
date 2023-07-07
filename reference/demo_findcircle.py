# encoding:utf-8
 
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
img = cv2.imread('E:\\cv_pre2seg\\source\\moeda.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图像
 
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('img'), plt.xticks([]), plt.yticks([])
# hough transform  规定检测的圆的最大最小半径，不能盲目的检测，否则浪费时间空间
#                          灰度图像，只支持霍夫圆检测，1为原始大小2为原始大小的一半；圆心之间的最小距离；canny检测器的高阈值
#                          检测阶段的累加器阈值，越小，会增加不存在的圆，越大检测到的圆形越完美；检测的最小圆半径；最大圆半径
circle1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=30, minRadius=20, maxRadius=40)
circles = circle1[0, :, :]  # 提取为二维
circles = np.uint16(np.around(circles))  # 四舍五入，取整
for i in circles[:]:
    cv2.circle(img, (i[0], i[1]), i[2], (255, 0, 0), 1)  # 画圆
    # cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 3)  # 画圆心
 
plt.subplot(122), plt.imshow(img)
plt.title('circle'), plt.xticks([]), plt.yticks([])
plt.show()