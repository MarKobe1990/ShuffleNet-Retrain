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
    def __init__(self, txt_name, data_path, transform, size, droup_out_class_label=None):
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
            if (droup_out_class_label is None) or (int(information[1]) not in droup_out_class_label):
                images.append(information[0])
                labels.append(int(information[1]))
        self.images = images
        self.labels = labels
        self.transform = transform
        # self.read_img = OpenCVResize(size=size)
        self.data_path = data_path

    # 重写这个函数用来进行图像数据的读取
    def __getitem__(self, item):
        # 获取图像名和标签
        image = self.images[item]
        # image = self.read_img(Image.open(self.data_path + image))
        image = Image.open(self.data_path + image)
        label = self.labels[item]
        # 处理图像数据
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.images)


def set_data_loader(dataset_attr_word, batch_size=10, size=256, shuffle=True, transforms_compose=None,
                    droup_out_class_label=None, dataset_dir=None):
    class_txt_path_dic = {
        "train": ['data_class_txt/train_classes.txt', dataset_dir + '/train/'],
        "val_easy": ['data_class_txt/val_easy_classes.txt', dataset_dir + '/test/COME15K-Easy/'],
        "val_hard": ['data_class_txt/val_hard_classes.txt', dataset_dir + '/test/COME15K-Hard/'],
        "test_easy": ['data_class_txt/test_easy_classes.txt', dataset_dir + '/test/COME15K-Easy/'],
        "test_hard": ['data_class_txt/test_hard_classes.txt', dataset_dir + '/test/COME15K-Hard/'],
    }
    dataset_attr = class_txt_path_dic.get(dataset_attr_word)
    dataset = COME15KDataSet(txt_name=dataset_attr[0], data_path=dataset_attr[1], transform=transforms_compose,
                             size=size, droup_out_class_label=droup_out_class_label)
    dataset_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, drop_last=True,
                                pin_memory=True)
    return dataset_loader


if __name__ == '__main__':
    train_transforms_compose = transforms.Compose([
        transforms.ToTensor()
        # RGB,imageNet1k mean and standard
    ])
    train_loader = set_data_loader("train", transforms_compose=train_transforms_compose)
    print(type(train_loader))
    for step, batch in enumerate(train_loader):
        for idx, (output_img, output_label) in enumerate(batch):
            print('step is :', idx)
            print('img is {}, label is {}'.format(output_img.shape, output_label.shape))
