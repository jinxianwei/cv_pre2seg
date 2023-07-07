import cv2
import numpy as np
 
img = cv2.imread('E:\\cv_pre2seg\\source\\moeda.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,50,150,apertureSize = 3)
lines = cv2.HoughLinesP(edges,1,np.pi/180,30,minLineLength=10,maxLineGap=10)
for line in lines:
    x1,y1,x2,y2 = line[0]
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
 
cv2.imshow('lines', img)
cv2.waitKey()
cv2.destroyAllWindows()