import numpy as np
import cv2 as cv
from trigo import *

class MyBall:

    def __init__(self, w, h, wg, hg):
        self.path = "image/game/ammo.png"
        self.w = w
        self.h = h
        self.wg = wg
        self.hg = hg
        self.img = cv.imread(self.path, cv.IMREAD_UNCHANGED)
        if self.img is None:
            print('Failed to load image file:', self.name)


        self.pos = Vector(320.0, 240.0)
        self.v = Vector(50.0, -50.0)
        self.touched = False

    def update(self, frame, t, track):
        self.pos.x += t * self.v.x * 10.0
        self.pos.y += t * self.v.y * 10.0
        self.contact_limit()
        self.contact_track(track)
        
        #print(str(self.pos.x) + " " + str(self.pos.y))

        for i in range(self.h):
            for j in range(self.w):
                for c in range(3):
                    if self.pos.x+j >= 0 and self.pos.x+j < self.wg and self.pos.y+i >= 0 and self.pos.y+i < self.hg:
                        if self.img[i, j, 3] != 0:
                            frame[int(self.pos.y)+i, int(self.pos.x)+j, c] = self.img[i, j, c]

        return frame
        
    def contact_limit(self):
        
        if self.pos.x < 0:
            self.v = reflect(self.v, normalize(Vector(1.0, 0.0)))
            self.pos.x = 0
        elif self.pos.x+self.w > self.wg:
            self.v = reflect(self.v, normalize(Vector(-1.0, 0.0)))
            self.pos.x = self.wg - self.w
        elif self.pos.y < 0:
            self.v = reflect(self.v, normalize(Vector(0.0, 1.0)))
            self.pos.y = 0
        elif self.pos.y +self.h > self.hg:
            self.v = reflect(self.v, normalize(Vector(0.0, -1.0)))
            self.pos.y = self.hg - self.h
        #print(str(self.v.x) + " " + str(self.v.y))
        
    def overlap(self, l1, r1, l2, r2):
        if l1.x == r1.x or l1.y == r1.y or r2.x == l2.x or l2.y == r2.y:
            return False
   
        if l1.x > r2.x or l2.x > r1.x:
            return False
 
        
        if r1.y > l2.y or r2.y > l1.y:
            return False
 
        return True

    def contact_track(self, track):
        bb = track.get_bounding_vox() 
        x = bb[0]
        y = bb[1]
        
        if distance(Vector(x+bb[2]//2, y+bb[3]//2), Vector(self.pos.x + self.w //2, self.pos.y + self.h // 2)) <= 100:
            if not self.touched:
                self.v = reflect(self.v, normalize(track.get_v()))
                #self.v  = Vector((4/3),(4/3)) * self.v + Vector(((1-2)/(2+1)),((1-2)/(2+1))) * track.get_v()
                self.touched = True
                print(str(track.get_v().x) + " " + str(track.get_v().y))
        else:
            self.touched = False
