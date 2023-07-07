import cv2  as cv
 
def cv_show(name, img):
    cv.imshow(name, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
 
img = cv.imread('E:\\cv_pre2seg\\source\\moeda.png')
if img is None:
    print("Failed to read the image")
 
# 沿x方向的边缘检测
img1 = cv.Sobel(img, cv.CV_64F, 1, 0)
sobelx = cv.convertScaleAbs(img1)
# 展示未进行取绝对值的图片
cv_show('img1', img1)
cv_show('sobelx', sobelx)
 
# 沿y方向的边缘检测
img1 = cv.Sobel(img, cv.CV_64F, 0, 1)
sobely = cv.convertScaleAbs(img1)
cv_show('sobely', sobely)
 
# 沿x，y方向同时检测，效果不是特别好
img1 = cv.Sobel(img, cv.CV_64F, 1, 1)
sobelxy = cv.convertScaleAbs(img1)
cv_show('sobelxy', sobelxy)
 
# 一般在x，y方向分别检测，在进行与运算
sobelxy1 = cv.bitwise_and(sobelx, sobely)
cv_show('sobelxy1', sobelxy1)
# 这种方法也行
 
sobelxy1 = cv.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
cv_show('sobelxy1', sobelxy1)
 