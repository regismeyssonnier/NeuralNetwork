import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2 as cv

Reseau_inceptionV3 = tf.keras.applications.InceptionV3()

FluxVideo = cv.VideoCapture(0)
if not FluxVideo.isOpened():
    print("No camera")
    exit()

while 1:
    readok, imgbgr = FluxVideo.read()

    if not readok:
        print("Probleme camera")
        break

    imgrgb = cv.cvtColor(imgbgr, cv.COLOR_BGR2RGB)
    imgbgr640x480 = cv.resize(imgbgr, (640, 480))

    cv.putText(imgbgr640x480, "space identifier esc sortir", (10,460), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3, cv.LINE_4)
     
    cv.imshow("Retour video", imgbgr640x480)

    Key = cv.waitKey(1)
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

    
        