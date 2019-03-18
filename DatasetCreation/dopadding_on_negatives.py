import cv2
# from matplotlib import pyplot as plt
import os
import glob

desired_size = 40

im_pth = "./raw_negatives"
os.chdir(im_pth)
images = glob.glob("*.png")

for img in images:
    print(img)
    im1 = cv2.imread(img, 0)
    old_size = im1.shape[:2]  # old_size is in (height, width) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])  # new_size should be in (width, height) format

    im = cv2.resize(im1, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    new_path = "/home/kartik/skylark/ImageClassifierDataset/negatives/" + img
    cv2.imwrite(new_path, new_im)

print("Padding Done")
