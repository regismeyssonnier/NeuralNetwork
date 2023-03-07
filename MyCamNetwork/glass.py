import numpy as np
import cv2 as cv


class MyGlass:

    def __init__(self):
        self.path = "image/glass.png"
               
        self.img = cv.imread(self.path, cv.IMREAD_UNCHANGED)
        if self.img is None:
            print('Failed to load image file:', self.name)
            

    def print_glasses(self, faces, eyes, frame):
        if len(faces) == 0:return frame
        #if len(eyes) < 2:return frame

        for (x, y, w, h) in faces:

            """x2,y2,w2,h2 = eyes[0]
            eye_center1 = (x + x2 + w2//2, y + y2 + h2//2)
            radius1 = int(round((w2 + h2)*0.25)) 

            x2,y2,w2,h2 = eyes[1]
            eye_center2 = (x + x2 + w2//2, y + y2 + h2//2)
            radius2 = int(round((w2 + h2)*0.25)) """

            wg = int(0.75*500)#int(500*(abs(eye_center1[0]-eye_center2[0])/500.0))*2
            if wg > 500:
                wg = 500
            hg = int(0.75*220)#int((float(wg) / 500.0) * (500.0/220.0) * 220.0)
            if hg < 100:
                hg = 100
            self.img = cv.resize(self.img, (wg, hg))
            #frame += self.img

            start = [0, 0]
            start[0] = x -(wg - w) // 2
            start[1] = y + 60
            for i in range(hg):
                for j in range(wg):
                    for c in range(3):
                        if start[0]+j >= 0 and start[0]+j < 640 and start[1]+i < 480:
                            if self.img[i, j, 3] != 0:
                                frame[start[1]+i, start[0]+j, c] = self.img[i, j, c]


        #print(str(w) + " " + str(h))
        return frame


#myglass = MyGlass()
#myglass.print_glasses([[75, 150, 200, 200]], [[50,50,50,50], [100, 50, 50, 50]], [])


