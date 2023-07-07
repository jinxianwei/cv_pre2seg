import cv2
import numpy

img = cv2.imread("source\moeda.png")
cv2.imshow("img", img)

# 1.图像二值化
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

kernel = numpy.ones((10, 10), dtype=numpy.uint8)
# 2.噪声去除
open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
# 3.确定背景区域
sure_bg = cv2.dilate(open, kernel, iterations=3)
# 4.寻找前景区域
dist_transform = cv2.distanceTransform(open, 1, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.9 * dist_transform.max(), 255, cv2.THRESH_BINARY)
# 5.找到未知区域
sure_fg = numpy.uint8(sure_fg)
unknow = cv2.subtract(sure_bg, sure_fg)

# 6.类别标记
ret, markers = cv2.connectedComponents(sure_fg)
# 为所有的标记加1，保证背景是0而不是1
markers = markers + 1
# 现在让所有的未知区域为0
markers[unknow == 255] = 0

# 7.分水岭算法
markers = cv2.watershed(img, markers)
img[markers == -1] = (0, 0, 255)

cv2.imshow("gray", gray)
cv2.imshow("thresh", thresh)
cv2.imshow("open", open)
cv2.imshow("sure_bg", sure_bg)
cv2.imshow("sure_fg", sure_fg)
cv2.imshow("unknow", unknow)
cv2.imshow("img_watershed", img)
cv2.waitKey(0)
cv2.destroyWindow()