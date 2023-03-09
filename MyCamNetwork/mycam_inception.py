import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv
from mynetwork import *
from me import *
from facedetection import *
from glass import *
from sobel import *
from filter import *
from tracker import *
from ball import *

Reseau_inceptionV3 = tf.keras.applications.InceptionV3()

FluxVideo = cv.VideoCapture(0)
if not FluxVideo.isOpened():
    print("No camera")
    exit()

myface = MyFace()
myglass = MyGlass()
mytracker = MyTracker(7)
myball = MyBall(50, 50, 640, 480)

face_active = False
its_regis = False
sobel = False
sobelinv = False
sobelfilter = 0
color_filter = -1
colorf = False
grabcut = False
fft = False
hough = False
canny = False
tracker = False
init_tracker = False
game = False

start_time = cv2.getTickCount()
while 1:
    
    readok, imgbgr = FluxVideo.read()

    if not readok:
        print("Probleme camera")
        break

    imgrgb = imgbgr#cv.cvtColor(imgbgr, cv.COLOR_BGR2RGB)
    imgbgr640x480 = cv.resize(imgrgb, (640, 480))
    cv.putText(imgbgr640x480, "space identifier esc sortir", (10,460), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv.LINE_4)
    if its_regis:
        cv.putText(imgbgr640x480, "C'est Regis", (10,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_4)

    if tracker:
        if not init_tracker:
            mytracker.select_roi(imgbgr640x480, 0, 0, 150, 100)
            init_tracker = True
        else:
            imgbgr640x480 = mytracker.update(imgbgr640x480)

    if game and tracker:
        end_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() 
        #print(end_time)
        #imgbgr640x480 = cv2.flip(imgbgr640x480, 1)
        myball.update(imgbgr640x480, end_time, mytracker)
        start_time = cv2.getTickCount()

    if grabcut:
        gc = GrabCutFilter(imgbgr640x480)
        imgbgr640x480 = gc.get_grabcut()

    if fft:
        fftf = FFTFilter(imgbgr640x480)
        imgbgr640x480 = fftf.get_fft()

    if sobel:
        sobelf = SobelFilter(imgbgr640x480)
        imgbgr640x480 = sobelf.get_sobel()

    if sobelinv:
        sobelf = SobelFilter(imgbgr640x480)
        imgbgr640x480 = sobelf.get_sobel_inv()

    if hough:
        h = HoughLFilter(imgbgr640x480)
        imgbgr640x480 = h.get_hough_line()

    if canny:
        h = CannyFilter(imgbgr640x480)
        imgbgr640x480 = h.get_edges()

    if colorf:
        if color_filter == 0:
            cf = ColorFilter(imgbgr640x480)
            imgbgr640x480 = cf.get_red_filter()
        elif color_filter == 1:
            cf = ColorFilter(imgbgr640x480)
            imgbgr640x480 = cf.get_green_filter()
        elif color_filter == 2:
            cf = ColorFilter(imgbgr640x480)
            imgbgr640x480 = cf.get_blue_filter()

    
    if face_active:
        imgbgr640x480detf = myface.detect(imgbgr640x480)
        img_gl = myglass.print_glasses(myface.get_faces(), myface.get_eyes(), imgbgr640x480detf)
        
        cv.imshow("Retour video", img_gl)
    else:
        cv.imshow("Retour video", imgbgr640x480)


    Key = cv.pollKey()
    #print(Key)
    if Key == 27:
        FluxVideo.release()

        cv.destroyAllWindows()
        break
    elif Key == 32:
        t1 = cv.getTickCount()
        
        imgrgb_299 = cv.resize(imgrgb, (299,299))
        imgrgb_299_dim4 = np.expand_dims(imgrgb_299, axis=0)

        entrees = tf.keras.applications.inception_v3.preprocess_input(imgrgb_299_dim4)

        sorties = Reseau_inceptionV3.predict(entrees)

        conclusions = tf.keras.applications.inception_v3.decode_predictions(sorties, top = 5)
        
        t2 =  cv.getTickCount()

        print("Conslusion : {:.2f} seconde".format((t2-t1)/cv.getTickFrequency()))

        #plt.imshow(imgrgb_299),plt.show()

        for i in range(5):
            synset_id = conclusions[0][i][1]
            confiance = "{:.2f}".format(conclusions[0][i][2])

            print(synset_id, confiance)
    elif Key == 97:

        cv.imwrite("image/copy/autre/" + str(np.random.randint(2000000000)) + ".jpg", imgrgb)

    elif Key == 114:#r
        #cv.imwrite("image/test.png", imgrgb)
        #if mynet.predict("image/test.png") == 1:
        imgrgb180x180 = cv.resize(imgrgb, (180, 180))
        if mynet.predict_mem(imgrgb180x180) == 1:
            its_regis = True
        else:
            its_regis = False
    elif Key == 112:
        cv.imwrite("image/copy/regis/" + str(np.random.randint(2000000000))+".jpg", imgrgb)
       
        
    elif Key == 102:#f
        face_active = not face_active
    elif Key == 103:#g
        game = not game

    elif Key == 115:#s
        sobel = not sobel
        sobelinv = False
    elif Key == 120:#x
        sobelinv = not sobelinv
        sobel = False

    elif Key == 110:#n
        sobelinv = False
        sobel = False
        colorf = False
        fft = False
        hough = False
        canny = False

    elif Key == 99:#c
        colorf = True
        color_filter = (color_filter+1)%3

    elif Key == 111:#o
        fft = not fft

    elif Key == 104: #h
        hough = not hough

    elif Key == 121: #y
        canny = not canny

    elif Key == 116: #t
        tracker = not tracker
        init_tracker = False