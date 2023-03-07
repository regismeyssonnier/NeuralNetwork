import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv
from mynetwork import *
from me import *
from facedetection import *
from glass import *

Reseau_inceptionV3 = tf.keras.applications.InceptionV3()

FluxVideo = cv.VideoCapture(0)
if not FluxVideo.isOpened():
    print("No camera")
    exit()

myface = MyFace()
myglass = MyGlass()
face_active = False
its_regis = False

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

        cv.imwrite("image/" + str(np.random.randint(2000000000)) + ".png", imgrgb)

    elif Key == 114:#r
        cv.imwrite("image/test.png", imgrgb)
        if mynet.predict("image/test.png") == 1:
            its_regis = True
        else:
            its_regis = False
    elif Key == 112:
        cv.imwrite("image/copy/regis/" + str(np.random.randint(2000000000))+".jpg", imgrgb)
       
        
    elif Key == 102:
        face_active = True
    elif Key == 103:
        face_active = False