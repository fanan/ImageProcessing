#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import scipy as sp
import cv2
from sklearn import svm

face_detector = cv2.CascadeClassifier("../data/haarcascade_frontalface_default.xml")

def standardizeImage(filename):
    img = cv2.imread(filename, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img = cv2.GaussianBlur(img, (3,3), 0)
    rects = face_detector.detectMultiScale(img)
    max_area = 0
    max_arg = 0
    i = 0
    for rect in rects:
        area = rect[2] * rect[3]
        if area > max_area:
            max_area = area
            max_arg = i
        i += 1
    rect = rects[max_arg]
    return cv2.resize(img[rect], (4,4)).ravel()

def gatherData():
    data = list()
    y = []
    for i in range(1,7):
        fn0 = "../data/wf/wf%d.jpeg" % i
        y.append(0)
        data.append(standardizeImage(fn0))
        fn1 = "../data/zzy/zzy%d.jpeg" % i
        y.append(1)
        data.append(standardizeImage(fn1))
    return data, y

if __name__ == '__main__':
    data, y = gatherData()
    #print data, y
    test_data1 = standardizeImage("../data/test/zzy.jpeg")
    clf = svm.SVC()
    clf.fit(data, y)
    print clf
    print clf.predict(test_data1)
    #print test_data
    test_data2 = standardizeImage("../data/test/wf.jpeg")
    print clf.predict(test_data2)
