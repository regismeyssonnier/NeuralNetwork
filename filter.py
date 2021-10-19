import numpy as np
import cv2

def filter_image(imgGray):

	#prepare the 5x5 shaped filter
	"""kernel = np.array([[1, 1, 1, 1, 1], 
		           [1, 1, 1, 1, 1], 
		           [1, 1, 1, 1, 1], 
		           [1, 1, 1, 1, 1], 
		           [1, 1, 1, 1, 1]])
	"""
	kernelBH = np.array([[1, 1, 1], 
		           [0, 3, 0], 
		           [-1, -1, -1]])

	kernelHB = np.array([[-1, -1, -1], 
		           [0, 3, 0], 
		           [1, 1, 1]])

	kernelGD = np.array([[-1, 0, 1], 
		             [-1, 3, 1], 
		             [-1, 0, 1]])

	kernelDG = np.array([[1, 0, -1], 
		             [1, 3, -1], 
		             [1, 0, -1]])

	kernelDBHD = np.array([[0, 1, 0], 
		              [1, 3, -1], 
		              [0, -1, 0]])

	kernelDBHG = np.array([[0, 1, 0], 
		              [-1, 3, 1], 
		              [0, -1, 0]])

	kernelDHBG = np.array([[0, -1, 0], 
		              [-1, 3, 1], 
		              [0, 1, 0]])

	kernelDHBD = np.array([[0, -1, 0], 
		              [1, 3, -1], 
		              [0, 1, 0]])

	#kernel = kernel/sum(kernel)

	#filter the source image
	img_rstbh = cv2.filter2D(imgGray,-1,kernelBH)
	"""img_rsthb = cv2.filter2D(img_rstbh,-1,kernelHB)
	img_rstgd = cv2.filter2D(img_rsthb,-1,kernelGD)
	img_rstdg = cv2.filter2D(img_rstgd,-1,kernelDG)

	img_rstdbhd = cv2.filter2D(img_rstdg,-1,kernelDBHD)
	img_rstdbhg = cv2.filter2D(img_rstdbhd,-1,kernelDBHG)
	img_rstdhbg = cv2.filter2D(img_rstdbhg,-1,kernelDHBG)
	img_rstdhbd = cv2.filter2D(img_rstdhbg,-1,kernelDHBD)"""
	
	#print(img_rstdhbd[0,0])

	return img_rstbh

def pooling_image(img, w, h, szk):
	I = 0
	imr = np.zeros((h/szk, w/szk), dtype=np.uint8)
	for i in range(0, h, szk):
		J = 0
		for j in range(0, w, szk):
			#----------------------
			mini = 1000
			for k in range(szk):
				for l in range(szk):
					if ((i+k) < h) and ((j+l) < w):
						if img[i+k, j+l] < mini:
							mini = img[i+k, j+l]
			
			if ((i+k) < h) and ((j+l) < w):				
				imr[I, J] = mini
			J +=1
		I += 1
		
	return imr

def pooling_image2(img, w, h, szk):
	I = 0
	imr = np.zeros(((h/szk)-2, (w/szk)-2), dtype=np.uint8)
	for i in range(0, h):
		J = 0
		for j in range(0, w):
			#----------------------
			mini = 1000
			for k in range(szk):
				for l in range(szk):
					if ((i+k) < h) and ((j+l) < w):
						if img[i+k, j+l] < mini:
							mini = img[i+k, j+l]
			
			if ((i+k) < h) and ((j+l) < w):				
				imr[I, J] = mini
			J +=1
		I += 1
		
	return imr

"""
#read image
img_src = cv2.imread('dog7.jpg')
#img_src = cv2.imread('pays.jpg')
imres = cv2.resize(img_src, (512, 512))
imgGray = cv2.cvtColor(imres, cv2.COLOR_BGR2GRAY)

imr = filter_image(imgGray)
imp = pooling_image(imr, 512, 512, 32)
imr2 = filter_image(imp)
imp2 = pooling_image(imr2, 256, 256, 2)
imr3 = filter_image(imp2)
imp3 = pooling_image(imr3, 128, 128, 2)
imr4 = filter_image(imp3)
imp4 = pooling_image(imr4, 64, 64, 2)
imr5 = filter_image(imp4)
imp5 = pooling_image(imr5, 32, 32, 2)


#save result image
cv2.imwrite('result.jpg',imp)
"""



