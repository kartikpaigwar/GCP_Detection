import cv2
# from matplotlib import pyplot as plt
import os
import glob
import numpy as np

desired_size = 40

im_pth = "./raw_positives"
os.chdir(im_pth)
images = glob.glob("*.png")

for img in images:
    print(img)
    im1 = cv2.imread(img, 0)
    old_size = im1.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size) / max(old_size)
    if ratio > 1:
        ratio = 0.65
    new_size = tuple([int(x * ratio) for x in old_size])  # new_size should be in (width, height) format

    im = cv2.resize(im1, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    hor_img = cv2.flip(new_im, 0)
    ver_img = cv2.flip(new_im, 1)

    rows, cols = new_im.shape
    theta = np.random.randint(10,350)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
    rot_img = cv2.warpAffine(new_im, M, (cols, rows))

    path = "/home/kartik/skylark/ImageClassifierDataset/positivesresize/resized"
    new_path = path + img
    cv2.imwrite(new_path, new_im)
    new_path = path + "hflip-"+ img
    cv2.imwrite(new_path, hor_img)
    new_path = path + "vflip-" + img
    cv2.imwrite(new_path, ver_img)
    new_path = path + "rot-" + img
    cv2.imwrite(new_path, rot_img)

print("Padding Done")
