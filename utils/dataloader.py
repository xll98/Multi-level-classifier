
import os
import random

import numpy as np
import torch
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data.dataset import Dataset

from .utils import cvtColor, preprocess_input, resize_image


def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a


def MMdataset_collate(batch):
    # 对于每个样本，将图像和标签分别存入列表
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)

    # 将图像列表转为numpy数组
    images = np.array(images)
    # 将图像转为PyTorch的Tensor类型，并设置为float类型
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    # 将标签转为PyTorch的long类型
    labels = torch.from_numpy(np.array(labels)).long()

    # 返回图像和标签
    return images, labels


class MMDataset(Dataset):
    def __init__(self, input_shape, lines, random):
        self.input_shape = input_shape
        self.lines = lines
        self.length = len(lines)
        self.random = random

        # ------------------------------------#
        #   路径和标签
        # ------------------------------------#
        self.paths = []
        self.level1_labels = []
        self.level2_labels = []
        self.level3_labels = []

        self.load_dataset()

    def __getitem__(self, index):
        # 获取指定索引的图像和标签

        image = np.zeros((1, 3, self.input_shape[0], self.input_shape[1]))
        labels = np.zeros((3))

        i = index % self.length
        image = cvtColor(Image.open(self.paths[i]))
        if self.rand() < .5 and self.random:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image = resize_image(image, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)
        image = preprocess_input(np.array(image, dtype='float32'))
        image = np.transpose(image, [2, 0, 1])
        labels[0] = self.level1_labels[i]
        labels[1] = self.level2_labels[i]
        labels[2] = self.level3_labels[i]
        return image, labels

    def __len__(self):
        return self.length
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    def load_dataset(self):
        for path in self.lines:
            path_split = path.split(";")
            self.paths.append(path_split[3].split()[0])
            self.level1_labels.append(int(path_split[0]))
            self.level2_labels.append(int(path_split[1]))
            self.level3_labels.append(int(path_split[2]))
        self.paths = np.array(self.paths, dtype=np.object)
        self.level1_labels = np.array(self.level1_labels)
        self.level2_labels = np.array(self.level2_labels)
        self.level3_labels = np.array(self.level3_labels)