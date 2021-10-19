import cv2
import sys
from filter import *

TEST = []
TEST_out = []
SIZE_I = 16

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

	
	img = cv2.imread(name)
	print(name)
	if img is None:
		print('Failed to load image file:', name)
		sys.exit(1)
	imres = cv2.resize(img, (512, 512))
	imgGray = cv2.cvtColor(imres, cv2.COLOR_BGR2GRAY)
	
	"""imr = filter_image(imgGray)
	imp = pooling_image(imr, 50, 50, 2)
	imr2 = filter_image(imp)
	imp2 = pooling_image(imr2, 25, 25, 2)"""

	imr = filter_image(imgGray)
	imp = pooling_image(imr, 512, 512, 32)
	"""imr2 = filter_image(imp)
	imp2 = pooling_image(imr2, 256, 256, 2)
	imr3 = filter_image(imp2)
	imp3 = pooling_image(imr3, 128, 128, 2)
	imr4 = filter_image(imp3)
	imp4 = pooling_image(imr4, 64, 64, 2)
	imr5 = filter_image(imp4)
	imp5 = pooling_image(imr5, 32, 32, 2)"""
	cv2.imwrite(name+".jpg",imp)
	return imp

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
				imf.append(im[i, j]/255) 

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

def load_image_one(nm, num, nb):

	croix = []
	name = ""
	numero = []

	
	name = nm + str(nb) + ".png"
	
	im = load_image(name)

	imf = []
	for i in range(SIZE_I):
		for j in range(SIZE_I):
			imf.append(1-(im[i, j]/255)) 

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
	
	TEST = []
	TEST_out = []

	

	#TEST_out[1] = 3
	#TEST_out[2] = 3

	t, n = load_image_lot("croix", 1, 5)
	for im in t:
		TEST.append(im)
	for to in n:
		TEST_out.append(to)	

	t, n = load_image_lot("trait", 2, 5)
	for im in t:
		TEST.append(im)
	for to in n:
		TEST_out.append(to)

	t, n = load_image_lot("cercle", 3, 5)
	for im in t:
		TEST.append(im)
	for to in n:
		TEST_out.append(to)

	


	return TEST, TEST_out


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

def create_valid_tab():
	t, n = load_image_lot("valid/cercle", 3, 18)
	testv = []
	

	for im in t:
		testv.append(im)
	

	return testv

def create_valid_tab_filter():
	
	testv = []
	
	t, n = load_image_lot_filter("archive/dog vs cat/dataset/test_set/cats/cat", 3, 20)
	for im in t:
		testv.append(im)

	t, n = load_image_lot_filter("archive/dog vs cat/dataset/test_set/dogs/dog", 3, 20)
	for im in t:
		testv.append(im)
	
	"""t, n = load_image_lot("valid/cercle", 3, 15)
	for im in t:
		testv.append(im)"""

	return testv
	

#TEST, TEST_out = create_test_tab()
TESTV = create_valid_tab_filter()

"""
print(len(TEST))
display_image(TEST[5])

l = ""
for t in TEST_out:
	l += str(t) + "  "
print(l)
"""

#cv2.imshow("Image", im)
#cv2.waitKey()
