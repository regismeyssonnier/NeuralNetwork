import cv2
import sys

TEST = []
TEST_out = []

def load_image(name):

	
	img = cv2.imread(name)
	print(name)
	if img is None:
		print('Failed to load image file:', name)
		sys.exit(1)

	imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#ret,mask = cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

	return imgGray
	

def load_image_lot(nm, num):

	croix = []
	name = ""
	numero = []

	for i in range(5):
		name = nm + str(i+1) + ".png"
		
		im = load_image(name)

		imf = []
		for i in range(50):
			for j in range(50):
				imf.append(im[i, j]/255) 

		croix.append(imf)
		numero.append(num)

	return croix, numero

def display_image(im):

	
	for i in range(50):
		l = ""
		for j in range(50):
			l+= str(im[i*50 + j]) + " "
		print(l) 

	
		
def create_test_tab():	
	t, n = load_image_lot("croix", 1)
	TEST = []
	TEST_out = []

	print("len"+str(len(t)))
	for im in t:
		TEST.append(im)
	for to in n:
		TEST_out.append(to)

	t, n = load_image_lot("trait", 2)
	for im in t:
		TEST.append(im)
	for to in n:
		TEST_out.append(to)

	return TEST, TEST_out

TEST, TEST_out = create_test_tab()

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
