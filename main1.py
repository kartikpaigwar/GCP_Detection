import cv2
import numpy as np
import csv
from scipy import ndimage
import glob, os
import pandas as pd
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

train_on_gpu = torch.cuda.is_available()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer
        self.conv1 = nn.Conv2d(1, 16, 5)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(1568*4, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 2)
        self.softmax = nn.LogSoftmax(dim=1)

        nn.init.xavier_uniform_(self.conv1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.conv2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc1.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.fc2.weight,
                                gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.fc3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(-1, 1568*4)
        x = F.relu(self.fc1(x))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.softmax(self.fc3(x))
        return x


# create a complete CNN
model = Net()
model.load_state_dict(torch.load('model1_99%.pt'))
model.eval()

if train_on_gpu:
    model.cuda()



def ispositive_example(roi):

    roi_arr = np.ascontiguousarray(roi, dtype=np.float32) / 255
    data = torch.from_numpy(roi_arr)
    data = data.unsqueeze(0)
    data = data.unsqueeze(0).cuda()
    # if train_on_gpu:
    #     data.cuda()
    output = model(data)
    _, pred = torch.max(output, 1)
    class_belongs = pred.data.cpu().numpy()
    return class_belongs[0]


def resize_roi(roi):

    desired_size = 40
    old_size = roi.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)

    if ratio > 1:
        ratio = 1

    new_size = tuple([int(x * ratio) for x in old_size])  # new_size should be in (width, height) format

    resized_roi = cv2.resize(roi, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    resized_roi = cv2.copyMakeBorder(resized_roi, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return resized_roi


def locate_gcp(cnt):
    # color = [0, 0, 0]
    # roi = cv2.copyMakeBorder(roi, 10, 10, 10, 10, cv2.BORDER_CONSTANT,value=color)
    # im2, contours, hierarchy = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[0]

    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)

    max_d = 0
    max_f = 0

    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        if d > max_d:
            max_d = d
            max_f = f
    far = tuple(cnt[max_f][0])
    # far = tuple([far[0]-10, far[1]-10])
    return far


cv2.namedWindow('GCP_detection',cv2.WINDOW_NORMAL)
# cv2.namedWindow('morphimage',cv2.WINDOW_NORMAL)
# cv2.namedWindow('greythresimage',cv2.WINDOW_NORMAL)

os.chdir("./CV-Assignment-Dataset")
output_path = "/home/ivlabs/users/karthik/gcp_detection/CV-Assignment-Output/"

csv_data = [['FileName','GCPLocation']]
for file in glob.glob("*.JPG"):
    #if file == "DSC01590.JPG":
    csv_row = []
    gcp_list = []
    print(file)
    img = cv2.imread(file)

    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)         #Grayscale conversion
    graythres = cv2.inRange(grayimg, 242, 255)              #Threshold in grayscale space
    kernel = np.ones((3, 3), np.uint8)                      #Perfom opening and closing

    closing = cv2.morphologyEx(graythres, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    im2, contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   #find contours
    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)                  #fitting standard bounding rectangle on the contours
        a = max(w, h)
        b = min(w, h)
        if a / b <= 1.8:                                    #filter Rectangles having aspect ration more than 3:2
            k = cv2.isContourConvex(cnt)                    #Check convexity of the contours
            if k is False:                                  #filter based on convexity
                if w*h > 9:                                  #filter minute rectangles based on area threshold
                    roi = closing[y:y+h, x:x+w]
                    roi_dilated = cv2.dilate(roi, kernel, iterations=1)
                    padded_roi = resize_roi(roi)
                    if ispositive_example(padded_roi) == 1:

                        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        gcp_point = locate_gcp(cnt)
                        print(gcp_point)
                        cv2.circle(img, gcp_point, 3, [0, 0, 255], -1)

                        gcp = [gcp_point[0],gcp_point[1]]
                        gcp_list.append(gcp)

                        if len(csv_row) ==0:
                            csv_row.append(file)

    if len(csv_row) != 0:
        csv_row.append(gcp_list)
        csv_data.append(csv_row)


    #cv2.imshow('GCP_detection',img)
    #cv2.waitKey(500)

    filepath = output_path + file
    cv2.imwrite(filepath,img)

with open(output_path+'gcp_locations_output.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(csv_data)

csvFile.close()

print("CSV creation Done")
