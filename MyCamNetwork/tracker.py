import numpy as np
import cv2 
from trigo import *

class MyTracker:

    def __init__(self, type):
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

        tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
        tracker_type = tracker_types[type]
 
        if int(minor_ver) < 3:
            tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                self.tracker = cv2.TrackerBoosting_create()
            elif tracker_type == 'MIL':
                self.tracker = cv2.TrackerMIL_create()
            elif tracker_type == 'KCF':
                self.tracker = cv2.TrackerKCF_create()
            elif tracker_type == 'TLD':
                self.tracker = cv2.TrackerTLD_create()
            elif tracker_type == 'MEDIANFLOW':
                self.tracker = cv2.TrackerMedianFlow_create()
            elif tracker_type == 'GOTURN':
                self.tracker = cv2.TrackerGOTURN_create()
            elif tracker_type == 'MOSSE':
                self.tracker = cv2.TrackerMOSSE_create()
            elif tracker_type == "CSRT":
                self.tracker = cv2.TrackerCSRT_create()

        self.lpos = Vector(0, 0)
        self.v = Vector(0,0)

    def select_roi(self, frame, x, y, w, h):

        self.bounding_box = (x, y, w, h)


        ok = self.tracker.init(frame, self.bounding_box)
        if not ok:
            print("bad initialization tracker !!!")

    def select_roi2(self, frame):

        self.bounding_box = cv2.selectROI(frame, False)


        ok = self.tracker.init(frame, self.bounding_box)
        if not ok:
            print("bad initialization tracker !!!")

    def get_bounding_vox(self):
        return self.bounding_box

    def get_v(self):
        return self.v

    def update(self, frame):

        timer = cv2.getTickCount()
        self.lpos = Vector(self.bounding_box[0], self.bounding_box[1])
        
        # Update tracker
        ok, self.bounding_box = self.tracker.update(frame)
        self.v = Vector(self.bounding_box[0] - self.lpos.x, self.bounding_box[1] - self.lpos.y)

        # Draw bounding box
        if ok:
            # Tracking success
            x = self.bounding_box[0] + self.bounding_box[2] // 2
            y = self.bounding_box[1] + self.bounding_box[3] // 2
            cv2.circle(frame, (x, y), 100, (255,0,0), 2)
        
        return frame