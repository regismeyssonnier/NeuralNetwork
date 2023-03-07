#coding=utf-8
import cv2
import math 
import numpy as np
# Charger l'image en niveaux de gris
#img = cv2.imread('archive/test/chihuahua/chihuahua.321.jpg', 0)
img = cv2.imread('head2.png', 0)

# Appliquer le filtre de Sobel pour obtenir les gradients en X et Y
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
gradient = np.sqrt(np.square(sobelx) + np.square(sobely))

# Normaliser le gradient pour avoir des valeurs entre 0 et 255
sx = cv2.normalize(sobelx, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
sy = cv2.normalize(sobely, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

gradient_norm = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
gradient_norm = 255 - gradient_norm 
# Seuiller le gradient pour enlever le fond
_, thresh = cv2.threshold(gradient_norm, 50, 255, cv2.THRESH_BINARY)
#imgf = cv2.resize(gradient_norm, (16,16))

# Afficher les images r?sultantes
cv2.imshow('Original', img)
cv2.imshow('Sobel X', sx)
cv2.imshow('Sobel Y', sy)
cv2.imshow('gradient', gradient_norm)
cv2.imshow('seuillage', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
