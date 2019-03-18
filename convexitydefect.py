import cv2
import numpy as np
import os
import glob

im_pth = "./ImageClassifierDataset/raw_positives/"
os.chdir(im_pth)
images = glob.glob("*.png")


for imgfile in images:
    img = cv2.imread(imgfile)
    color = [0, 0, 0]
    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT,
                                     value=color)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #ret, thresh = cv2.threshold(img_gray, 254, 255,0)
    im2, contours, hierarchy = cv2.findContours(img_gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]

    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)

    max_d = 0
    max_s = 0
    max_e = 0
    max_f = 0

    for i in range(defects.shape[0]):
         s,e,f,d = defects[i,0]
         if d > max_d:
             max_d = d
             max_s = s
             max_e = e
             max_f = f

    start = tuple(cnt[max_s][0])
    end = tuple(cnt[max_e][0])
    far = tuple(cnt[max_f][0])
    #cv2.line(img,start,end,[0,255,0],2)
    cv2.circle(img,far,2,[0,0,255],-1)
    cv2.imshow('img',img)
    cv2.waitKey(500)
