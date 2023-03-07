import cv2
import sys
import numpy as np


class Me:

    def __init__(self, sn):
        self.surname = sn


    def load_image(self, name):
        self.name = name
        self.img = cv2.imread(name)
        print("loading image: " + name)
        if self.img is None:
            print('Failed to load image file:', self.name)
            sys.exit(1)
        

    def create_copy_random(self, path, nb, num_start, name, ext, inter):
        print(self.img.shape)
        for i in range(nb):
            img_copy = np.copy(self.img)   
            img_rd = np.random.randint(inter[0], inter[1], size=img_copy.shape)
            for j in range(img_copy.shape[0]):
                for k in range(img_copy.shape[1]):
                    for l in range(img_copy.shape[2]):
                        img_copy[j,k,l] += img_rd[j, k, l]
            img_copy = np.clip(img_copy, 0, 255)
            cv2.imwrite(path + name + str(num_start+i) + ext, img_copy)
            print(path + name + str(num_start+i) + ext + " created.")

"""
me = Me("regis")
me.load_image("image/head2.png")
me.create_copy_random("image/copy/", 50, 1, "regis", ".png", (-5, 5))
me.load_image("image/rstraight.png")
me.create_copy_random("image/copy/", 50, 51, "regis", ".png", (-1, 1))
me.load_image("image/rleft.png")
me.create_copy_random("image/copy/", 50, 101, "regis", ".png", (-1, 1))
me.load_image("image/head2.png")
me.create_copy_random("image/copy/", 50, 251, "regis", ".png", (-50, 50))
"""