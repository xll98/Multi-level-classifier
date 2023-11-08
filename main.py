'''
Author: mircle mircle@mircle.com
Date: 2023-11-03 14:10:59
LastEditors: mircle mircle@mircle.com
LastEditTime: 2023-11-03 15:01:08
FilePath: \PaimianClass\main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from utils.dataloader import MMdataset_collate, MMDataset
from torch.utils.data import DataLoader
from utils.utils import show_config, get_num_list
import numpy as np
import torch

if __name__ == "__main__":
    # -------------------------------------------------------#
    #   指向根目录下的cls_train.txt，读取人脸路径与标签
    # --------------------------------------------------------#
    annotation_path = "cls_train.txt"
    # annotation_path = "cls_train.txt.back"
    # --------------------------------------------------------#
    #   输入图像大小，常用设置如[112, 112, 3]
    # --------------------------------------------------------#
    # input_shape     = [80, 144, 3]
    input_shape = [160, 160, 3]
    batch_size = 27
    # -------------------------------------------------------#
    #   0.01用于验证，0.99用于训练
    #   验证集的比例
    # -------------------------------------------------------#
    val_split = 0.15
    with open(annotation_path, "r", encoding='utf-8') as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    train_dataset = MMDataset(input_shape, lines[:num_train], random=True)
    val_dataset = MMDataset(input_shape, lines[num_train:], random=False)

    train_sampler = None
    val_sampler = None
    shuffle = True

    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=1, pin_memory=True,
                     drop_last=True, collate_fn=MMdataset_collate, sampler=train_sampler)
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=1, pin_memory=True,
                         drop_last=True, collate_fn=MMdataset_collate, sampler=val_sampler)

    count_nums = 0
    count_batchs = 0
    for iteration, batch in enumerate(gen):  # 遍历训练数据集
        images, labels = batch  # 获取图像和标签
        count_nums += 1
        count_batchs += images.shape[0]
        print("image shape", images.shape)
        print("labels.shape", labels.shape)
        labels_l1 = labels[:, 0].view(-1, 1)
        labels_l2 = labels[:, 1]
        labels_l3 = labels[:, 2]
        print("labels_l1.shape", labels_l1.shape)
        print("labels[0]", labels[0])
        print("labels_l1[0]", labels_l1[0])
        break
    class_num1, class_num2, class_num3 = get_num_list(annotation_path)
    print("num1",class_num1)
    print("num2", class_num2)
    print("num3", class_num3)
