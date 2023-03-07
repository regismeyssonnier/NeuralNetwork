import numpy as np
import cv2 as cv


class MyFace:

    def __init__(self):
        self.face_data = "data/frontface.xml"
        self.eye_data = "data/eye.xml"
        
        self.face_cascade = cv.CascadeClassifier()
        self.eyes_cascade = cv.CascadeClassifier()
        #-- 1. Load the cascades
        if not self.face_cascade.load(cv.samples.findFile(self.face_data)):
            print('--(!)Error loading face cascade')

        if not self.eyes_cascade.load(cv.samples.findFile(self.eye_data)):
            print('--(!)Error loading eyes cascade')
         
        self.eyes = []
        self.faces=[]


    def get_eyes(self):
        return self.eyes

    def get_faces(self):
        return self.faces

    def detect(self, frame):

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)
        #-- Detect faces
        self.faces = self.face_cascade.detectMultiScale(frame_gray)
        for (x,y,w,h) in self.faces:
            """center = (x + w//2, y + h//2)
            frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)"""
            faceROI = frame_gray[y:y+h,x:x+w]
            #-- In each face, detect eyes
            self.eyes = self.eyes_cascade.detectMultiScale(faceROI)
            """for (x2,y2,w2,h2) in self.eyes:
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                radius = int(round((w2 + h2)*0.25))
                frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)"""
        

        return frame