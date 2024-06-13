
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
    def __init__(self, path, im_dim) -> None:
        super().__init__()
        self.path = path
        self.im_list = list(set(img_name[:12] for img_name in listdir(path) if img_name[-3:] == "png"))
        self.resize = Resize(im_dim, interpolation=InterpolationMode.NEAREST)
        # self.normalize = Normalize((0.5,),  (0.5,))
        self.colors = np.array(labels_colors)
        self.one_hot = np.identity(32)
    
    def __getitem__(self, index):
        im_name = self.path + "/" + self.im_list[index] + ".png"
        lb_name = self.path + "/" + self.im_list[index] + "_OH.png"

        image = read_image(im_name).to(float32)
        image = self.resize(image)

        label = onehot_mask(lb_name, labels_colors)
        label = self.resize(label)

        return image, label

    def __len__(self):
        return len(self.im_list)


dataset = MyDataset("dataset/CamSeq01", (800, 800))
train_set, test_set = random_split(dataset, [90, 11])
train_set, val_set = random_split(train_set, [80, 10])

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=10)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

if __name__ == "__main__":
    res = train_set[0]
    pass