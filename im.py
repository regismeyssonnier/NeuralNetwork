import cv2
import sys
from filter import *
from mix import *
import numpy as np
import tensorflow as tf


TEST = []
TEST_out = []
SIZE_I = 16
MAX_IMG_SQ = 2
IMG = 0

def load_image(name):
	
	
	img = cv2.imread(name)
	print(name)
	if img is None:
		print('Failed to load image file:', name)
		sys.exit(1)

	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#ret,mask = cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	return imgGray

def load_image_filter(name):

	global IMG
	img = cv2.imread(name)
	print(name + " " + str(IMG))
	IMG+=1
	if img is None:
		print('Failed to load image file:', name)
		sys.exit(1)
	imres = cv2.resize(img, (256,256))
	#print(imres)
	#imgGray = cv2.cvtColor(imres, cv2.COLOR_BGR2GRAY)
	#imgrgb = cv2.cvtColor(imres, cv2.COLOR_BGR2RGB)
	#cv2.imwrite("archive/test/result/"+ str(np.random.randint(2000000000)) + ".jpg" ,imgGray)
	#ret, thresh = cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#ret, thresh = cv2.threshold(imgGray, 150, 255, cv2.THRESH_BINARY)
	#img_cont = np.zeros((128, 128), dtype=np.uint8)
	#contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
	#cv2.drawContours(image=img_cont, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
	"""sobelx = cv2.Sobel(imres, cv2.CV_64F, 1, 0, ksize=5)
	sobely = cv2.Sobel(imres, cv2.CV_64F, 0, 1, ksize=5)
	gradient = np.sqrt(np.square(sobelx) + np.square(sobely))

	# Normaliser le gradient pour avoir des valeurs entre 0 et 255
	gradient_norm = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
	
	# Convertir en niveaux de gris avec la methode de la luminance
	gray_luminance = cv2.cvtColor(gradient_norm, cv2.COLOR_RGB2GRAY)

	imgf = pooling_image(gray_luminance, 256, 256, 16)"""

	imgr = cv2.cvtColor(imres, cv2.COLOR_RGB2GRAY)
	imgf = pooling_image(imgr, 256,256, 16)
	#print(imgf.shape)
	
	"""
	imr = filter_image(imgGray)
	imp = pooling_image(imr, 512, 512, 2)
	imr2 = filter_image(imp)
	imp2 = pooling_image(imr2, 256, 256, 2)
	imr3 = filter_image(imp2)
	imp3 = pooling_image(imr3, 128, 128, 2)
	imr4 = filter_image(imp3)
	imp4 = pooling_image(imr4, 64, 64, 2)
	imr5 = filter_image(imp4)
	imp5 = pooling_image(imr5, 32, 32, 2)"""
	#"archive/test/result/"
	#cv2.imwrite("archive/test/result/"+ str(np.random.randint(2000000000)) + ".jpg" ,imgf)
	return imgf

def load_image_filterCrC(name):

	global IMG
	img = cv2.imread(name)
	print(name + " " + str(IMG))
	IMG+=1
	if img is None:
		print('Failed to load image file:', name)
		sys.exit(1)
	#imres = cv2.resize(img, (256,256))
	#print(imres)
	#imgGray = cv2.cvtColor(imres, cv2.COLOR_BGR2GRAY)
	#imgrgb = cv2.cvtColor(imres, cv2.COLOR_BGR2RGB)
	#cv2.imwrite("archive/test/result/"+ str(np.random.randint(2000000000)) + ".jpg" ,imgGray)
	#ret, thresh = cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#ret, thresh = cv2.threshold(imgGray, 150, 255, cv2.THRESH_BINARY)
	#img_cont = np.zeros((128, 128), dtype=np.uint8)
	#contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
	#cv2.drawContours(image=img_cont, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
	"""sobelx = cv2.Sobel(imres, cv2.CV_64F, 1, 0, ksize=5)
	sobely = cv2.Sobel(imres, cv2.CV_64F, 0, 1, ksize=5)
	gradient = np.sqrt(np.square(sobelx) + np.square(sobely))

	# Normaliser le gradient pour avoir des valeurs entre 0 et 255
	gradient_norm = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
	
	# Convertir en niveaux de gris avec la methode de la luminance
	gray_luminance = cv2.cvtColor(gradient_norm, cv2.COLOR_RGB2GRAY)

	imgf = pooling_image(gray_luminance, 256, 256, 16)"""

	imgr = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	#imgf = pooling_image(imgr, 256,256, 16)
	#print(imgf.shape)
	
	"""
	imr = filter_image(imgGray)
	imp = pooling_image(imr, 512, 512, 2)
	imr2 = filter_image(imp)
	imp2 = pooling_image(imr2, 256, 256, 2)
	imr3 = filter_image(imp2)
	imp3 = pooling_image(imr3, 128, 128, 2)
	imr4 = filter_image(imp3)
	imp4 = pooling_image(imr4, 64, 64, 2)
	imr5 = filter_image(imp4)
	imp5 = pooling_image(imr5, 32, 32, 2)"""
	#"archive/test/result/"
	#cv2.imwrite("archive/test/result/"+ str(np.random.randint(2000000000)) + ".jpg" ,imgf)
	return imgr

def load_image_filter_k(name):

	global IMG
	"""img = cv2.imread(name)
	print(name)
	if img is None:
		print('Failed to load image file:', name)
		sys.exit(1)
	imres = cv2.resize(img, (64, 64))
	imgr = cv2.cvtColor(imres, cv2.COLOR_RGB2GRAY)
	Xim = np.array([imgr])"""
	#img_cont = np.zeros((258, 258), dtype=np.uint8)
	#print(imres)
	#imres = imres.reshape((258, 258, 3))
	#print(imres.shape)
	print(name + " " + str(IMG))
	IMG+=1
	img = tf.keras.utils.load_img(
		name, target_size=(256, 256)
	)
	img_array = tf.keras.utils.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0)

	model = tf.keras.models.Sequential([
	
		tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(4, 11, activation='relu', strides=(1,1), padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(8, 7, activation='relu', strides=(1,1), padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, 5, activation='relu', strides=(1,1), padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu', strides=(1,1), padding="same"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu', strides=(1,1), padding="same"),
    tf.keras.layers.MaxPooling2D(),
		
	#tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    #tf.keras.layers.MaxPooling2D((2, 2)),

    
    tf.keras.layers.Flatten()

	])
	
	"""tf.keras.layers.Rescaling(1./255),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(32, 3, activation='relu'),
                tf.keras.layers.MaxPooling2D(),"""
	

	imp = model.predict(img_array)
	#print(imp)
	#model.summary()
	#print(imp)
	#model.summary()
	#print(imp.shape)
	impr = np.zeros(4096, dtype=np.float)
	ind = 0
	"""for i in range(16):
		for j in  range(16):
			impr[i, j] = imp[0][ind]
			ind +=1"""

	for i in range(4096):
		impr[i] = imp[0][i]
			

	#cv2.imwrite("archive/test/result/"+ str(np.random.randint(2000000000)) + ".jpg" ,impr)
	return impr
	

def load_image_lot(nm, num, nb):

	croix = []
	name = ""
	numero = []

	for i in range(nb):
		name = nm + str(i+1) + ".png"
		
		im = load_image(name)

		imf = []
		for i in range(SIZE_I):
			for j in range(SIZE_I):
				imf.append(float(im[i, j]/255)) 

		croix.append(imf)
		numero.append(num)

	return croix, numero

def load_image_lot_filter(nm, num, nb):

	croix = []
	name = ""
	numero = []

	for i in range(nb):
		name = nm + str(i+1) + ".jpg"
		
		im = load_image_filter(name)

		imf = []
		for i in range(SIZE_I):
			for j in range(SIZE_I):
				#print(im[i, j])
				imf.append(float(float(im[i, j])/float(255.0))) 

		croix.append(imf)
		numero.append(num)

	return croix, numero

def load_image_lot_filter_png(nm, num, nb):

	croix = []
	name = ""
	numero = []

	for i in range(nb):
		name = nm + str(i+1) + ".png"
		
		im = load_image_filter(name)

		imf = []
		for i in range(SIZE_I):
			for j in range(SIZE_I):
				#print(im[i, j])
				imf.append(float(float(im[i, j])/float(255.0))) 

		croix.append(imf)
		numero.append(num)

	return croix, numero

def load_image_one(nm, num, nb):

	croix = []
	name = ""
	numero = []

	
	name = nm + str(nb) + ".png"
	
	im = load_image(name)

	imf = []
	for i in range(SIZE_I):
		for j in range(SIZE_I):
			imf.append(float(im[i, j]/255)) 

	croix.append(imf)
	numero.append(num)

	return croix, numero

def load_image_one_filter(nm, num, nb):

	croix = []
	name = ""
	numero = []

	
	name = nm + str(nb) + ".png"
	
	im = load_image_filter(name)

	imf = []
	for i in range(SIZE_I):
		for j in range(SIZE_I):
			imf.append(float(im[i, j]/255)) 

	croix.append(imf)
	numero.append(num)

	return croix, numero

def load_image_one_filter_rand(nm, num):

	croix = []
	name = ""
	numero = []
	
	
	im = load_image_filter(nm)

	imf = []
	for i in range(SIZE_I): 
		for j in range(SIZE_I): 
			pix = float(1.0- (im[i, j]/255.0)) 
			#pix = float(im[i, j])
			#if pix >= 0.5:pix = 1.0
			#else:pix = 0.0
			imf.append(pix)   

	croix.append(imf)
	numero.append(num)

	return croix, numero

def load_image_one_CrC(nm, num):

	croix = []
	name = ""
	numero = []
	
	
	im = load_image_filterCrC(nm)

	imf = []
	for i in range(SIZE_I): 
		for j in range(SIZE_I): 
			pix = float(1.0 - (im[i, j]/255.0)) 
			#pix = float(im[i, j])
			#if pix >= 0.5:pix = 1.0
			#else:pix = 0.0
			imf.append(pix)   

	croix.append(imf)
	numero.append(num)

	return croix, numero

def load_image_one_filter_rand_rgb(nm, num):

	croix = []
	name = ""
	numero = []
	
	
	im = load_image_filter(nm)

	imf = []
	for i in range(SIZE_I):
		for j in range(SIZE_I):
			for k in range(3):
				pix = float(im[i, j, k]/255.0)
				#if pix >= 0.5:pix = 1.0
				#else:pix = 0.0
				imf.append(pix) 

	croix.append(imf)
	numero.append(num)

	return croix, numero

def display_image(im):

	
	for i in range(SIZE_I):
		l = ""
		for j in range(SIZE_I):
			l+= str(im[i*SIZE_I + j]) + " "
		print(l) 

	
		
def create_test_tab():	
	
	test = []
	test_out = []

	

	#TEST_out[1] = 3
	#TEST_out[2] = 3

	t, n = load_image_lot("croix", 1, 5)
	for im in t:
		test.append(im)
	for to in n:
		test_out.append(to)	

	t, n = load_image_lot("trait", 2, 5)
	for im in t:
		test.append(im)
	for to in n:
		test_out.append(to)

	t, n = load_image_lot("cercle", 3, 5)
	for im in t:
		test.append(im)
	for to in n:
		test_out.append(to)

	


	return test, test_out

def create_test_tab_filter():	
	
	test = []
	test_out = []

	

	#TEST_out[1] = 3
	#TEST_out[2] = 3

	t, n = load_image_lot_filter("archive/dog vs cat/dataset/training_set/cats/cat", 1, 20)
	for im in t:
		test.append(im)
	for to in n:
		test_out.append(to)	

	"""t, n = load_image_lot("cercle", 2, 15)
	for im in t:
		test.append(im)
	for to in n:
		test_out.append(to)"""

	t, n = load_image_lot_filter("archive/dog vs cat/dataset/training_set/dogs/dog.", 2, 20)
	for im in t:
		test.append(im)
	for to in n:
		test_out.append(to)

	"""t, n = load_image_lot_filter("black", -1, 1)
	for im in t:
		test.append(im)
	for to in n:
		test_out.append(to)"""

	"""t, n = load_image_lot_filter("cercle", 3, 5)
	for im in t:
		test.append(im)
	for to in n:
		test_out.append(to)"""

	


	return test, test_out

def create_test_tab_one():	
	
	test = []
	test_out = []

	"""for i in range(5):
		t, n = load_image_one("croix", 1, i+1)
		test.append(t[0])
		test_out.append(n[0])

		t, n = load_image_one("trait", 2, i+1)
		test.append(t[0])
		test_out.append(n[0])

		t, n = load_image_one("cercle", 3, i+1)
		test.append(t[0])
		test_out.append(n[0])"""

	

	t, n = load_image_one("croix", 1, 1)
	test.append(t[0])
	test_out.append(n[0])
	
	t, n = load_image_one("croix", 1, 2)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one("cercle", 3, 1)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one("trait", 2, 3)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one("cercle", 3, 5)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one("croix", 1, 5)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one("cercle", 3, 3)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one("trait", 2, 2)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one("trait", 2, 5)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one("croix", 1, 4)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one("cercle", 3, 2)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one("trait", 2, 1)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one("croix", 1, 3)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one("cercle", 3, 4)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one("trait", 2, 4)
	test.append(t[0])
	test_out.append(n[0])


	return test, test_out

def create_test_tab_one_filter():	
	
	test = []
	test_out = []

	"""for i in range(5):
		t, n = load_image_one("croix", 1, i+1)
		test.append(t[0])
		test_out.append(n[0])

		t, n = load_image_one("trait", 2, i+1)
		test.append(t[0])
		test_out.append(n[0])

		t, n = load_image_one("cercle", 3, i+1)
		test.append(t[0])
		test_out.append(n[0])"""

	

	t, n = load_image_one_filter("croix", 1, 1)
	test.append(t[0])
	test_out.append(n[0])
	
	t, n = load_image_one_filter("croix", 1, 2)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("cercle", 3, 1)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("trait", 2, 3)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("cercle", 3, 5)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("croix", 1, 5)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("cercle", 3, 3)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("trait", 2, 2)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("trait", 2, 5)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("croix", 1, 4)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("cercle", 3, 2)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("trait", 2, 1)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("croix", 1, 3)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("cercle", 3, 4)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("trait", 2, 4)
	test.append(t[0])
	test_out.append(n[0])


	return test, test_out

def create_test_tab_one_filter_cercle():	
	
	test = []
	test_out = []

	"""for i in range(5):
		t, n = load_image_one("croix", 1, i+1)
		test.append(t[0])
		test_out.append(n[0])

		t, n = load_image_one("trait", 2, i+1)
		test.append(t[0])
		test_out.append(n[0])

		t, n = load_image_one("cercle", 3, i+1)
		test.append(t[0])
		test_out.append(n[0])"""

	

	t, n = load_image_one_filter("croix", 1, 1)
	test.append(t[0])
	test_out.append(n[0])
	
	t, n = load_image_one_filter("croix", 1, 2)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("cercle", 3, 1)
	test.append(t[0])
	test_out.append(n[0])

	#t, n = load_image_one_filter("trait", 2, 3)
	#test.append(t[0])
	#test_out.append(n[0])

	t, n = load_image_one_filter("cercle", 3, 5)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("croix", 1, 5)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("cercle", 3, 3)
	test.append(t[0])
	test_out.append(n[0])

	#t, n = load_image_one_filter("trait", 2, 2)
	#test.append(t[0])
	#test_out.append(n[0])

	#t, n = load_image_one_filter("trait", 2, 5)
	#test.append(t[0])
	#test_out.append(n[0])

	t, n = load_image_one_filter("croix", 1, 4)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("cercle", 3, 2)
	test.append(t[0])
	test_out.append(n[0])

	#t, n = load_image_one_filter("trait", 2, 1)
	#test.append(t[0])
	#test_out.append(n[0])

	t, n = load_image_one_filter("croix", 1, 3)
	test.append(t[0])
	test_out.append(n[0])

	t, n = load_image_one_filter("cercle", 3, 4)
	test.append(t[0])
	test_out.append(n[0])

	#t, n = load_image_one_filter("trait", 2, 4)
	#test.append(t[0])
	#test_out.append(n[0])


	return test, test_out

def create_valid_tab():
	t, n = load_image_lot("valid/cercle", 3, 18)
	testv = []
	

	for im in t:
		testv.append(im)
	

	return testv

def create_valid_tab_filter():
	
	testv = []
	testv_out = []
	
	t, n = load_image_lot_filter("archive/dog vs cat/dataset/test_set/dogs/dog", 2, 20)
	for im in t:
		testv.append(im)
	for o in n:
		testv_out.append(o)
		
		
	t, n = load_image_lot_filter("archive/dog vs cat/dataset/test_set/cats/cat", 1, 20)
	for im in t:
		testv.append(im)
	for o in n:
		testv_out.append(o)

		
	"""t, n = load_image_lot("valid/cercle", 3, 15)
	for im in t:
		testv.append(im)"""

	return testv, testv_out
	
def create_test_tab_one_filter_rand_valid():	

    test = []
    test_out = []
    path = []
    pn = []
    
    path, pn = load_mix_image_valid()
    print("/////////////////////" + str(MAX_I))
    for i in range(MAX_I):
        t, n = load_image_one_filter_rand(path[i], pn[i])
        test.append(t[0])
        test_out.append(n[0])


    return test, test_out, path

def create_test_tab_one_filter_rand_train():	

    test = []
    test_out = []
    path = []
    pn = []
    
    path, pn = load_mix_image_train()
    print("/////////////////////" + str(MAX_I))
    for i in range(MAX_I):
        t, n = load_image_one_filter_rand(path[i], pn[i])
        test.append(t[0])
        test_out.append(n[0])


    return test, test_out, path

def create_test_tab_one_filter_rand_square_valid():	

    test = []
    test_out = []
    path = []
    pn = []
    
    path, pn = load_img_square()
    print("/////////////////////" + str(MAX_IMG_SQ))
    for i in range(MAX_IMG_SQ):
        t, n = load_image_one_filter_rand(path[i], pn[i])
        test.append(t[0])
        test_out.append(n[0])


    return test, test_out, path

def create_test_tab_one_filter_rand_square_train():	

    test = []
    test_out = []
    path = []
    pn = []
    
    path, pn = load_img_square()
    print("/////////////////////" + str(MAX_IMG_SQ))
    for i in range(MAX_IMG_SQ):
        t, n = load_image_one_filter_rand(path[i], pn[i])
        test.append(t[0])
        test_out.append(n[0])


    return test, test_out, path

def create_test_tab_one_filter_rand_batch_valid(num):	

    test1 = []
    test_out1 = []
    test2 = []
    test_out2 = []
    path1 = []
    pn1 = []
    path2 = []
    pn2 = []
    
    path1, pn1, path2, pn2 = load_img_batch(num)
    print("/////////////////////" + str(num))
    for i in range(num):
        t, n = load_image_one_filter_rand(path1[i], pn1[i])
        test1.append(t[0])
        test_out1.append(n[0])

    for i in range(num):
        t, n = load_image_one_filter_rand(path2[i], pn2[i])
        test2.append(t[0])
        test_out2.append(n[0])


    return test1, test_out1, path1, test2, test_out2, path2

def create_test_tab_one_filter_rand_batch_train(num):	

    test1 = []
    test_out1 = []
    test2 = []
    test_out2 = []
    path1 = []
    pn1 = []
    path2 = []
    pn2 = []
    
    path1, pn1, path2, pn2 = load_img_batch50(num)
    print("/////////////////////" + str(num))
    for i in range(num):
        t, n = load_image_one_filter_rand(path1[i], pn1[i])
        test1.append(t[0])
        test_out1.append(n[0])

    for i in range(num):
        t, n = load_image_one_filter_rand(path2[i], pn2[i])
        test2.append(t[0])
        test_out2.append(n[0])


    return test1, test_out1, path1, test2, test_out2, path2

def Normalize_img_batch(TEST, bsz):
    eps=1e-5
    batch_size = bsz
    n_batches = len(TEST) // batch_size
    if len(TEST) % batch_size != 0:
        n_batches += 1
    X_batches = np.array_split(TEST, n_batches)

    # Normaliser chaque mini-batch
    for i in range(n_batches):
        # Calcule la moyenne et la variance de chaque mini-batch
        batch_mean = np.mean(X_batches[i])
        batch_var = np.var(X_batches[i])
    
        # Normalise chaque mini-batch
        X_batches[i] = (X_batches[i] - batch_mean) / np.sqrt(batch_var + eps)

    # Fusionner les mini-batchs normalisés
    X_normalized = np.concatenate(X_batches, axis=0)

    return X_normalized

def create_test_tab_one_filter_rand_batch_train3(num):	
    global IMG
    IMG=0
    test1 = []
    test_out1 = []
    test2 = []
    test_out2 = []
    test3 = []
    test_out3 = []

    path1 = []
    pn1 = []
    path2 = []
    pn2 = []
    path3 = []
    pn3 = []
    
    path1, pn1, path2, pn2, path3, pn3 = load_img_batch_3(num)
    print("/////////////////////" + str(num))
    for i in range(num):
        t, n = load_image_one_filter_rand(path1[i], pn1[i])
        test1.append(t[0])
        test_out1.append(n[0])

    for i in range(num):
        t, n = load_image_one_filter_rand(path2[i], pn2[i])
        test2.append(t[0])
        test_out2.append(n[0])

    for i in range(num):
        t, n = load_image_one_filter_rand(path3[i], pn3[i])
        test3.append(t[0])
        test_out3.append(n[0])


    return test1, test_out1, path1, test2, test_out2, path2, test3, test_out3, path3


def create_test_tab_one_filter_rand_CrC(num):	
    global IMG
    IMG=0
    test1 = []
    test_out1 = []
    test2 = []
    test_out2 = []
    test3 = []
    test_out3 = []

    path1 = []
    pn1 = []
    path2 = []
    pn2 = []
    path3 = []
    pn3 = []
    
    path1, pn1, path2, pn2 = load_img_batch_CrC(num)
    print("/////////////////////" + str(num))
    for i in range(num):
        t, n = load_image_one_CrC(path1[i], pn1[i])
        test1.append(t[0])
        test_out1.append(n[0])

    for i in range(num):
        t, n = load_image_one_CrC(path2[i], pn2[i])
        test2.append(t[0])
        test_out2.append(n[0])

   

    return test1, test_out1, path1, test2, test_out2, path2

#for image dog and cat

##TESTV, TESTV_out = create_valid_tab_filter()
#end for

#cercle croix
#TEST, TEST_out = create_test_tab_one_filter_cercle();
#TESTV, TESTV_out = create_test_tab_one_filter_cercle();


"""
print(len(TEST))
for im in TEST:
	display_image(im)

l = ""
for t in TEST_out:
	l += str(t) + "  "
print(l)

"""
#cv2.imshow("Image", im)
#cv2.waitKey()
