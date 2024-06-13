
from torch import float32, Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from torchvision.transforms import Resize, Normalize, InterpolationMode
from os import listdir
import numpy as np

from dataset.onehot import onehot_mask


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
labels_colors = [[color] for color in range(32)]

class MyDataset(Dataset):
    def __init__(self, path, im_dim, test=False, val=False) -> None:
        super().__init__()
        self.path = path

        if test:
            self.folder = "test"
        elif val:
            self.folder = "val"
        else:
            self.folder = "train"

        self.im_list = list(set(img_name[:-4] for img_name in listdir("{}/{}".format(path, self.folder)) if img_name[-3:] == "png"))
        self.resize = Resize(im_dim, interpolation=InterpolationMode.NEAREST)
        # self.normalize = Normalize((0.5,),  (0.5,))
        self.colors = np.array(labels_colors)
        self.one_hot = np.identity(32)
    
    def __getitem__(self, index):
        im_name = self.path + "/{}/".format(self.folder) + self.im_list[index] + ".png"
        lb_name = self.path + "/{}_labels/".format(self.folder) + self.im_list[index] + "_OH.png"

        image = read_image(im_name).to(float32)
        image = self.resize(image)
        # image = self.normalize(image)

        label = onehot_mask(lb_name, labels_colors)
        label = self.resize(label)

        return image, label

    def __len__(self):
        return len(self.im_list)


im_size = (800, 800)
train_set = MyDataset("dataset/CamVid", im_size)
test_set = MyDataset("dataset/CamVid", im_size, test=True)
val_set = MyDataset("dataset/CamVid", im_size, val=True)

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=10)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

if __name__ == "__main__":
    res = train_set[0]
    pass
