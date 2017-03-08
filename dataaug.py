import cv2
import random
import pickle
import constants as c
import numpy as np


class DataAug():
    def translate_x(self, img):
        rows, cols = img.shape
        jitter = random.uniform(-c.Xtranslate, c.Xtranslate)
        M = np.float32([[1, 0, jitter], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    def translate_y(self, img):
        rows, cols = img.shape
        jitter = random.uniform(-c.Ytranslate, c.Ytranslate)
        M = np.float32([[1, 0, 0], [0, 1, jitter]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        return dst

    def rotateImage(self, image):
        jitter = random.uniform(-c.Rotate, c.Rotate)
        image_center = tuple(np.array(image.shape) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, jitter, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape, flags=cv2.INTER_LINEAR)
        return result

    def addJitters(self, image):
        timage = self.translate_x(image)
        timage = self.translate_y(timage)
        timage = self.rotateImage(timage)
        timage = np.expand_dims(timage, axis=0)
        return timage

    def dataaug(self, image, label):
        tlabel = np.array([label])
        timage = self.addJitters(image)

        for i in range(1, c.NUM_JITTERS):
            temp = self.addJitters(image)
            timage = np.append(timage, temp, axis=0)
            tlabel = np.append(tlabel, np.array([label]))
        return timage,tlabel

"""
img = pickle.load(open("pickled_image.p", "rb"))
da = DataAug()

s, t = da.dataaug(img,0)

print(t.shape)
"""