import numpy as np
import cv2

img = cv2.imread('./source/2.jpg')
cv2.imshow('gude.png',img)

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

cv2.imshow("Deteccao de bordas com o Watershed",img)
cv2.waitKey(0)