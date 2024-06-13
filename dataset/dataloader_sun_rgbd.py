
from torch import float32, Tensor, uint8
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.io import read_image
from torchvision.transforms import Resize, Normalize, InterpolationMode
from os import listdir
import numpy as np
from scipy.io import loadmat


class MyDataset(Dataset):
    def __init__(self, path, num_classes, im_dim) -> None:
        super().__init__()
        self.path = path
        self.num_classes = num_classes
        self.im_list_1 = list(set(img_name for img_name in listdir(path+"/b3dodata") if img_name[0] != "."))
        # self.im_list_2 = list(set(img_name for img_name in listdir(path+"/NYUdata") if img_name[0] != "."))
        self.resize = Resize(im_dim, interpolation=InterpolationMode.NEAREST)
        # self.normalize = Normalize((0.5,),  (0.5,))
        self.one_hot = np.identity(32)
    
    def __getitem__(self, index):
        # if index < len(self.im_list_1):
        #     im_name = self.path + "/b3dodata/" + self.im_list_1[index] + "/image/" + self.im_list_1[index] + ".jpg"
        #     lb_name = self.path + "/b3dodata/" + self.im_list_1[index] + "/seg.mat"
        # else:
        #     index -= len(self.im_list_1)
        #     im_name = self.path + "/NYUdata/" + self.im_list_2[index] + "/image/" + self.im_list_2[index] + ".jpg"
        #     lb_name = self.path + "/NYUdata/" + self.im_list_2[index] + "/seg.mat"
        
        im_name = self.path + "/b3dodata/" + self.im_list_1[index] + "/image/" + self.im_list_1[index] + ".jpg"
        lb_name = self.path + "/b3dodata/" + self.im_list_1[index] + "/seg.mat"

        image = read_image(im_name).to(float32)
        image = self.resize(image)
        # image = self.normalize(image)

        label = loadmat(lb_name)
        names = label["names"]
        label = label["seglabel"]

        label = np.stack([label == i for i in range(self.num_classes)], axis=0).astype(np.uint8)
        label = Tensor(label)
        label = self.resize(label)

        return image, names.shape

    def __len__(self):
        return len(self.im_list_1)

dataset = MyDataset("dataset/SUNRGBD/SUNRGBD/kv1", 256, (224, 224))
train_set, test_set = random_split(dataset, [500, 54])
train_set, val_set = random_split(train_set, [400, 100])

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1)
test_loader = DataLoader(test_set, batch_size=1, shuffle=True)

if __name__ == "__main__":
    res = train_set[0]
    print(res[1].shape)
    pass