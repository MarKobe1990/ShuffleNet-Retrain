import torch
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision.transforms import transforms


class OpenCVResize(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img)  # (H,W,3) RGB
        img = img[:, :, ::-1]  # 2 BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size / H * W + 0.5), self.size) if H < W else (self.size, int(self.size / W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1]  # 2 RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img


# class ToBGRTensor(object):
#
#     def __call__(self, img):
#         assert isinstance(img, (np.ndarray, PIL.Image.Image))
#         if isinstance(img, PIL.Image.Image):
#             img = np.asarray(img)
#         img = img[:, :, ::-1]  # 2 BGR
#         img = np.transpose(img, [2, 0, 1])  # 2 (3, H, W)
#         img = np.ascontiguousarray(img)
#         img = torch.from_numpy(img).float()
#         return img

class COME15KDataSet(Dataset):
    def __init__(self, txt_name, data_path, transform, size):
        super(COME15KDataSet, self).__init__()
        # 打开存储图像名与标签的txt文件
        fp = open(txt_name, 'r')
        images = []
        labels = []
        # 将图像名和图像标签对应存储起来
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0])
            labels.append(int(information[1]))
        self.images = images
        self.labels = labels
        self.transform = transform
        self.read_img = OpenCVResize(size=size)
        self.data_path = data_path

    # 重写这个函数用来进行图像数据的读取
    def __getitem__(self, item):
        # 获取图像名和标签
        image = self.images[item]
        image = self.read_img(Image.open(self.data_path + image))
        label = self.labels[item]
        # 处理图像数据
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.images)


def set_data_loader(dataset_attr_word, batch_size=10, size=256, shuffle=True):
    class_txt_path_dic = {
        "train": ['data/SOD-SemanticDataset/train_classes.txt', 'data/SOD-SemanticDataset/train/'],
        "val_easy": ['data/SOD-SemanticDataset/val_easy_classes.txt', 'data/SOD-SemanticDataset/test/COME15K-Easy/'],
        "val_hard": ['data/SOD-SemanticDataset/val_hard_classes.txt', 'data/SOD-SemanticDataset/test/COME15K-Hard/'],
        "test_easy": ['data/SOD-SemanticDataset/test_easy_classes.txt', 'data/SOD-SemanticDataset/test/COME15K-Easy/'],
        "test_hard": ['data/SOD-SemanticDataset/test_hard_classes.txt', 'data/SOD-SemanticDataset/test/COME15K-Hard/'],
    }
    transforms_compose = transforms.Compose([
        transforms.RandomResizedCrop(224),
        # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # RGB,imageNet1k mean and standard
    ])
    dataset_attr = class_txt_path_dic.get(dataset_attr_word)
    dataset = COME15KDataSet(txt_name=dataset_attr[0], data_path=dataset_attr[1], transform=transforms_compose,
                             size=size)
    dataset_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, drop_last=True)
    return dataset_loader


if __name__ == '__main__':
    train_loader = set_data_loader("train")
    print(type(train_loader))
    for idx, (output_img, output_label) in enumerate(train_loader):
        print('step is :', idx)
        print('img is {}, label is {}'.format(output_img.shape, output_label.shape))
