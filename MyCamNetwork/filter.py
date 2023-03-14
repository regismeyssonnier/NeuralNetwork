#coding=utf-8
import cv2 as cv
import math 
import numpy as np

class ColorFilter:

    def __init__(self, img):

        self.img = img


    def get_blue_filter(self):
        
        for i in range(480):
                for j in range(640):
                    self.img[i, j, 1] = 0
                    self.img[i, j, 2] = 0
   
        return self.img

    def get_green_filter(self):
        
        for i in range(480):
                for j in range(640):
                    self.img[i, j, 0] = 0
                    self.img[i, j, 2] = 0
   
        return self.img

    def get_red_filter(self):
        
        for i in range(480):
                for j in range(640):
                    self.img[i, j, 0] = 0
                    self.img[i, j, 1] = 0
   
        return self.img

class GrabCutFilter:

    def __init__(self, img):

        self.img = img

    def get_grabcut(self):
        
        mask = np.zeros(self.img.shape[:2],np.uint8)
        bgdModel = np.zeros((1,65),np.float64)
        fgdModel = np.zeros((1,65),np.float64)
        rect = (0,0,639,479)
        cv.grabCut(self.img,mask,rect,bgdModel,fgdModel,1,cv.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        self.img = self.img*mask2[:,:,np.newaxis]

        return self.img

class FFTFilter:

    def __init__(self, img):

        self.img = img

    def get_fft(self):

        f = np.fft.fft2(self.img)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20*np.log(np.abs(fshift))

        rows, cols, color = self.img.shape
        crow,ccol = rows//2 , cols//2
        fshift[crow-80:crow+80, ccol-80:ccol+80] = 0
        #fshift = np.zeros((rows, cols))
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.real(img_back)

        img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)

        return img_back


class HoughLFilter:

    def __init__(self, img):

        self.img = img

    def get_hough_line(self):

        gray = cv.cvtColor(self.img,cv.COLOR_RGB2GRAY)
        edges = cv.Canny(gray,50,200,apertureSize = 3)
        lines = cv.HoughLinesP(edges,1,np.pi/180,50, None,minLineLength=150,maxLineGap=10)
        if lines is None :return self.img
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv.line(self.img,(x1,y1),(x2,y2),(255,0,0),2, cv.LINE_AA)
        
        return self.img

class CannyFilter:

    def __init__(self, img):

        self.img = img

    def get_edges(self):

        gray = cv.cvtColor(self.img,cv.COLOR_RGB2GRAY)
        edges = cv.Canny(gray,50,200,apertureSize = 3)
        edges = cv.cvtColor(edges, cv.COLOR_GRAY2RGB)
                
        return edges