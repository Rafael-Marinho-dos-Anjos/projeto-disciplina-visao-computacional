
import numpy as np
from os import listdir
import cv2
from tqdm import tqdm


labels_colors = [
            (64, 128, 64),
            (192, 0, 128),
            (0, 128, 192),
            (0, 128, 64),
            (128, 0, 0),
            (64, 0, 128),
            (64, 0, 192),
            (192, 128, 64),
            (192, 192, 128),
            (64, 64, 128),
            (128, 0, 192),
            (192, 0, 64),
            (128, 128, 64),
            (192, 0, 192),
            (128, 64, 64),
            (64, 192, 128),
            (64, 64, 0),
            (128, 64, 128),
            (128, 128, 192),
            (0, 0, 192),
            (192, 128, 128),
            (128, 128, 128),
            (64, 128, 192),
            (0, 0, 64),
            (0, 64, 64),
            (192, 64, 128),
            (128, 128, 0),
            (192, 128, 192),
            (64, 0, 64),
            (192, 192, 0),
            (0, 0, 0),
            (64, 192, 0)
         ]



path = "dataset/CamVid/val_labels"
path = "dataset/CamVid/train_labels"
path = "dataset/CamVid/test_labels"
im_list = list(set(img_name[:-6] for img_name in listdir(path) if img_name[-6:] == "_L.png"))
im_done = list(set(img_name[:-7] for img_name in listdir(path) if img_name[-6:] == "OH.png"))

for image in tqdm(im_list):
    if image in im_done:
        continue

    img = cv2.imread(path + "/" + image + "_L.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    try:
        img2 = np.array([[labels_colors.index(tuple(color)) for color in line] for line in img], dtype=np.int8)
    except:
        print("Image {} skipped".format(image))
        continue

    cv2.imwrite(path + "/" + image + "_OH.png", img2)
