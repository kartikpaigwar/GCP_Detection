import cv2
import numpy as np
from scipy import ndimage
import glob, os
import pandas as pd

data = pd.read_csv('./MLDataset1/gcp_locations.csv')
files = data.FileName
locs = data.GCPLocation

def if_img_exist_in_csv(imgfile):
    index = 0
    for file in files:
        id = file.index('\\')
        filename = file[id + 1:]
        if filename == imgfile:
            loc = locs[index].replace("[", "")
            loc = loc.replace("]", "")
            loc = np.fromstring(loc, dtype=float, sep=',')
            loc = np.reshape(loc, (-1, 2))
            loc = loc.astype(dtype=int)
            return loc
        index+=1
    return None

def ispositive_example(x1,y1,x2,y2,gcp_loc):
    if gcp_loc is None:
        return False
    for gcp_point in gcp_loc:
        if gcp_point[0]>x1 and gcp_point[0]<x2 and gcp_point[1]>y1 and gcp_point[1]<y2 :
            return True



cv2.namedWindow('rgbthresimage',cv2.WINDOW_NORMAL)
cv2.namedWindow('morphimage',cv2.WINDOW_NORMAL)
cv2.namedWindow('greythresimage',cv2.WINDOW_NORMAL)

os.chdir("./MLDataset1")

exampleno = 0

debug = 0
for file in glob.glob("*.JPG"):
    # if file == "M1_F1.3_0460.JPG":
    print(file)
    img = cv2.imread(file)
    gcp_loc= if_img_exist_in_csv(file)

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         #Grayscale conversion
    graythres = cv2.inRange(grayimg, 242, 255)              #Threshold in grayscale space
    kernel = np.ones((3, 3), np.uint8)                      #Perfom opening and closing

    closing = cv2.morphologyEx(graythres, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    im2, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   #find contours
    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)                  #fitting standard bounding rectangle on the contours
        a = max(w, h)
        b = min(w, h)
        if a / b <= 1.8:                                    #filter Rectangles having aspect ration more than 3:2
            k = cv2.isContourConvex(cnt)                    #Check convexity of the contours
            if k is False:                                  #filter based on convexity
                if w*h >9:                                  #filter minute rectangles based on area threshold
                    exampleno +=1
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    roi = closing[y:y+h,x:x+w]
                    roi_dilated = cv2.dilate(roi, kernel, iterations=1)
                    if ispositive_example(x,y,x+w,y+h,gcp_loc) is True:
                        cv2.imwrite("/home/kartik/skylark/positives1/positive"+ file +"-" + str(exampleno) + ".png",roi )
                        cv2.imwrite("/home/kartik/skylark/positives1/positive-dilated-" + file +"-"+str(exampleno) + ".png", roi_dilated)
                    else:
                        cv2.imwrite("/home/kartik/skylark/negatives1/negative" + file + "-" + str(exampleno) + ".png",roi)
        # cv2.imshow('rgbthresimage',img)
        # cv2.waitKey(0)
print("Dataset creation Done")
