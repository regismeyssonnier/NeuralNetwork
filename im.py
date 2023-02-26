import cv2
import sys
from filter import *
from mix import *
import numpy as np

TEST = []
TEST_out = []
SIZE_I = 16
MAX_IMG_SQ = 2

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
	imres = cv2.resize(img, (64, 64))
	imgGray = cv2.cvtColor(imres, cv2.COLOR_BGR2GRAY)
	#cv2.imwrite("archive/test/result/"+ str(np.random.randint(2000000000)) + ".jpg" ,imgGray)
	ret, thresh = cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#ret, thresh = cv2.threshold(imgGray, 150, 255, cv2.THRESH_BINARY)
	img_cont = np.zeros((64, 64), dtype=np.uint8)
	contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
	cv2.drawContours(image=img_cont, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

	"""imr = filter_image(imgGray)
	imp = pooling_image(imr, 50, 50, 2)
	imr2 = filter_image(imp)
	imp2 = pooling_image(imr2, 25, 25, 2)"""

	#imr = filter_image(imgGray)
	imp = img_cont#pooling_image(img_cont, 512, 512, 32)
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
	cv2.imwrite("archive/test/result/"+ str(np.random.randint(2000000000)) + ".jpg" ,imp)
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
			pix = float(im[i, j]/255.0)
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

#for image dog and cat
TEST, TEST_out, PATH, TEST2, TEST_out2, PATH2 = create_test_tab_one_filter_rand_batch_train(50)
## TEST, TEST_out = create_test_tab_filter()
TESTV, TESTV_out, PATHV, TESTV2, TESTV_out2, PATHV2 = create_test_tab_one_filter_rand_batch_valid(50)
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
