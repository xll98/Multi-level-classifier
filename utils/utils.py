import numpy as np
import torch
from PIL import Image
import math
from functools import partial

# -*- coding: utf-8 -*-
# ---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
# ---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image

    # ---------------------------------------------------#


#   对输入图像进行resize,并减去均值
# ---------------------------------------------------#
def resize_image_use(image, size, letterbox_image):
    w, h = size
    new_image = image.resize((w, h), Image.BICUBIC)
    image_array = np.array(new_image, dtype=np.float32)
    mean_value = np.mean(image_array, axis=(0, 1))
    image_array -= mean_value
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)
    # 减均值
    #return Image.fromarray(image_array)
    # 不减均值
    return new_image


def resize_image(image, size, letterbox_image):
    iw, ih = image.size
    w, h = size
    if letterbox_image:
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image


def get_num_classes(annotation_path):
    with open(annotation_path, encoding='utf-8') as f:
        dataset_path = f.readlines()

    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes


def get_num_list_fromlabel(label1_path, label2_path, label3_path):
    list_l1 = read_list_from_file_1d(label1_path)
    list_l2 = read_list_from_file_2d(label2_path)
    list_l3 = read_list_from_file_3d(label3_path)
    l1_num_classes = len(list_l1)
    l2_num_classes = []
    l3_num_classes = []
    for cls1 in range(l1_num_classes):
        l2_tmp_classes = list_l2[cls1]
        l2_tmp_num_class = len(l2_tmp_classes)
        l2_num_classes.append(l2_tmp_num_class)
        l3_num_classes_tmp = []
        for cls2 in range(l2_tmp_num_class):
            l3_tmp_classes = list_l3[cls1][cls2]  # 获取第一级分类为cls1且第二级分类为cls2的第二级标签
            l3_tmp_num_class = len(l3_tmp_classes)
            l3_num_classes_tmp.append(l3_tmp_num_class)
        l3_num_classes.append(l3_num_classes_tmp)
    return l1_num_classes, l2_num_classes, l3_num_classes

def get_num_list(annotation_path):
    """
    从标注文件中读取标签信息并计算不同级别的类别数量。

    Args:
        annotation_path (str): 包含标签信息的文件路径。

    Returns:
        int: 第一级分类的总类别数。
        list of int: 每个第一级分类下的第二级分类总类别数的列表。
        list of list of int: 每个第一级分类下的每个第二级分类下的第三级分类总类别数的列表。
    """
    # 打开标注文件，逐行读取
    with open(annotation_path, encoding='utf-8') as f:
        dataset_path = f.readlines()

    # 初始化三个空列表，用于保存不同级别的标签
    labels1 = []
    labels2 = []
    labels3 = []

    # 对于每一行标注，按分号分割后，将不同级别的标签添加到对应的列表中
    for path in dataset_path:
        path_split = path.split(";")
        labels1.append(int(path_split[0]))
        labels2.append(int(path_split[1]))
        labels3.append(int(path_split[2]))

    # 将不同级别的标签转换为numpy数组
    labels1 = np.array(labels1)
    labels2 = np.array(labels2)
    labels3 = np.array(labels3)

    # 计算不同级别标签的类别数量
    l1_num_classes = np.max(labels1) + 1  # 第一级分类的总类别数
    l2_num_classes = []
    l3_num_classes = []
    for cls1 in range(l1_num_classes):
        l2_tmp_classes = labels2[labels1 == cls1]  # 获取第一级分类为cls1的第二级标签
        l2_tmp_num_class = np.max(l2_tmp_classes) + 1  # 每个第一级分类下的第二级分类总类别数
        l2_num_classes.append(l2_tmp_num_class)

        l3_num_classes_tmp = []
        for cls2 in range(l2_tmp_num_class):
            l3_tmp_classes = labels3[(labels1 == cls1) & (labels2 == cls2)]  # 获取第一级分类为cls1且第二级分类为cls2的第二级标签
            if len(l3_tmp_classes) == 0:
                l3_tmp_num_class = 1
            else:
                l3_tmp_num_class = np.max(l3_tmp_classes) + 1  # 每个第一级分类下的每个第二级分类下的第三级分类总类别数
            l3_num_classes_tmp.append(l3_tmp_num_class)
        l3_num_classes.append(l3_num_classes_tmp)

    # 返回结果
    return l1_num_classes, l2_num_classes, l3_num_classes


# ---------------------------------------------------#
#   获得学习率
# ---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def preprocess_input(image):
    image /= 255.0
    return image


def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.1, warmup_lr_ratio=0.1,
                     no_aug_iter_ratio=0.3, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
                                              ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                    1.0
                    + math.cos(
                math.pi
                * (iters - warmup_total_iters)
                / (total_iters - warmup_total_iters - no_aug_iter)
            )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def write_list_to_file_1d(lst, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for item in lst:
            file.write(item + '\n')


def read_list_from_file_1d(filename):
    lst = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            lst.append(line.strip())  # 使用strip()方法去除行末的换行符
    return lst

def write_list_to_file_2d(lst, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for sublist in lst:
            file.write(','.join(map(str, sublist)) + '\n')  # 使用逗号分隔每个子列表的元素，并添加换行符


def read_list_from_file_2d(filename):
    lst = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            sublist = line.strip().split(',')  # 使用逗号分隔每一行，得到子列表
            lst.append(list(map(str, sublist)))  # 将子列表中的元素转换为整数并添加到新的列表中
    return lst


def write_list_to_file_3d(lst, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for sublist1 in lst:
            for sublist2 in sublist1:
                for item in sublist2:
                    file.write(str(item) + ' ')  # 使用逗号分隔三级子列表元素
                file.write(';')
            file.write('\n')  # 换行分隔不同的一级子列表
def read_list_from_file_3d(filename):
    lst = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if len(line) > 1:
                sublist2 = []
                str_list = line.split(";")  # 先按分号分割字符串
                for item in str_list:
                    if len(item)>1:
                        num_list = [str(n) for n in item.split()]  # 再将每个子串转化为数字列表
                        sublist2.append(num_list)
                lst.append(sublist2)  # 将一级子列表添加到原始列表中
    return lst