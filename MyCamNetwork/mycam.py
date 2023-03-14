import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv
import threading
import queue
from mynetwork import *
from me import *
from facedetection import *
from glass import *
from sobel import *
from filter import *
from tracker import *
from ball import *
from mysound import *
from myaudio import *
from myrecognition import *
from myspeech import *

num_commande = ""
its_regis = False

def qui_cest():
    mysound = MySound("temp/qui.wav")
    mysound.record(3, 16000, 1)
    mysound.wait()
    mysound.save()
    r = myaudio.predict("./temp/qui.wav")
    if r == 2:
        imgrgb180x180 = cv.resize(imgrgb, (180, 180))
        if mynet.predict_mem(imgrgb180x180) == 1:
            mysound = MySound("./regis.wav")
            mysound.play()
            global its_regis
            its_regis = True


def ai_vocale():
    mysound = MySound("regis.wav")
    text = mysound.recognize()
    print("analyze = " + text)
    myreco = MyRecognition()
    myreco.set_text(text)
    text, num = myreco.analyze()
    print("analyze = " + text)
    global num_commande
    num_commande = num
    myspeech = MySpeech(text, "fr")
    myspeech.speak()
    
def get_command_ai():
    while True:
        pass


FluxVideo = cv.VideoCapture(0)
if not FluxVideo.isOpened():
    print("No camera")
    exit()

myface = MyFace()
myglass = MyGlass()
mytracker = MyTracker(7)
myball = MyBall(50, 50, 640, 480)

aivocal = False
qai = queue.Queue()
face_active = False

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
    cv.putText(imgbgr640x480, "b vocal esc sortir", (10,460), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv.LINE_4)
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
        #print("col " + str(color_filter))
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

    elif Key == 100:
        t1 = threading.Thread(target=qui_cest)
        t1.start()

    elif Key == 98:
        aivocal = True
        t1 = threading.Thread(target=ai_vocale)
        t1.start()

    if num_commande != "":
        print("num " + str(num_commande))
        if num_commande == "jeu":
            tracker = True
            game = True
        elif num_commande == "sobeln":
            sobel = True
            sobelinv = False
        elif num_commande == "sobelbl":
            sobel = True
            sobelinv = True
        elif num_commande == "colvert":
            colorf = True
            color_filter = 1
        elif num_commande == "colrouge":
            colorf = True
            color_filter = 0
        elif num_commande == "colbleu":
            colorf = True
            color_filter = 2

        num_commande = ""