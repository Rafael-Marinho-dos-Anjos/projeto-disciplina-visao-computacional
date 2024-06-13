
import numpy as np
from os import listdir
import cv2
from tqdm import tqdm
import numpy as np


folder_path = r"dataset\cityscapes_data\\"
labels_colors = None

for folder in ["train", "val"]:
    path = folder_path + str(folder)
    im_list = listdir(path)
    for image in tqdm(im_list):
        img = cv2.imread(path + "\\" + image)
        mask = img[:, img.shape[1]//2:, :]
        img = img[:, :img.shape[1]//2, :]
        cv2.imwrite(folder_path+"\\data\\"+image[:-3]+".png", img)
        cv2.imwrite(folder_path+"\\data\\"+image[:-3]+"_L.png", mask)

        if labels_colors is None:
            labels_colors = mask.reshape((-1, 3))
        else:
            labels_colors = np.concatenate((labels_colors, mask.reshape((-1, 3))), axis=0)

        labels_colors = np.unique(labels_colors, axis=0)

        print(labels_colors.shape)


for folder in range(1, 9):
    path = folder_path + str(folder) 
    im_list = listdir(path + r"\masks")
    for image in tqdm(im_list):
        img = cv2.imread(path + r"\masks\\" + image)
        shape_0 = img.shape[0] - img.shape[0] % 32
        shape_1 = img.shape[1] - img.shape[1] % 32
        if shape_0 > 256:
            img = cv2.resize(img, (256, 256), cv2.INTER_NEAREST)
        else:
            img = cv2.resize(img, (shape_1, shape_0), cv2.INTER_NEAREST)
        
        cv2.imwrite(r"dataset\Semantic segmentation dataset\data" + "/{}_mask_".format(folder) + image, img)
        
        img = cv2.imread(path + r"\images\\" + image[:-3] + "jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if shape_0 > 256:
            img = cv2.resize(img, (256, 256), cv2.INTER_AREA)
        else:
            img = cv2.resize(img, (shape_1, shape_0), cv2.INTER_AREA)

        cv2.imwrite(r"dataset\Semantic segmentation dataset\data" + "/{}_image_".format(folder) + image, img)
