#coding=utf-8
import cv2
import math 
import numpy as np

class SobelFilter:

    def __init__(self, img):
        self.img = img
    
    def get_sobel(self):

        # Charger l'image en niveaux de gris
        #img = cv2.imread('archive/test/chihuahua/chihuahua.321.jpg', 0)

        # Appliquer le filtre de Sobel pour obtenir les gradients en X et Y
        sobelx = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient = np.sqrt(np.square(sobelx) + np.square(sobely))

        gradient_norm = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
       
        return gradient_norm

    def get_sobelXY(self):
        sobelx = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        # Normaliser le gradient pour avoir des valeurs entre 0 et 255
        sx = cv2.normalize(sobelx, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        sy = cv2.normalize(sobely, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        return sx, sy


    def get_sobel_inv(self):

        sobelx = cv2.Sobel(self.img, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(self.img, cv2.CV_64F, 0, 1, ksize=5)
        gradient = np.sqrt(np.square(sobelx) + np.square(sobely))

        gradient_norm = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        gradient_norm = 255 - gradient_norm 

        return gradient_norm