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
import os
def rename_folders(path):
    for foldername in os.listdir(path):
        if os.path.isdir(os.path.join(path, foldername)):
            new_name = foldername.rpartition(' -')[0]
            os.rename(os.path.join(path, foldername), os.path.join(path, new_name))

if __name__ == "__main__":
    # 示例列表
    #my_list3d = [ [[111,112], [121], [131,132], [141,142]], [[21], [22], [23]], [[311,312], [321,322,323]]]
    #my_list2d = [[11, 12, 13, 14], [21, 22, 23], [31, 32]]
    # 将列表写入文件
    #write_list_to_file_3d(my_list3d, 'my_file.txt')
    #write_list_to_file_2d(my_list2d, 'my_file2d.txt')

    # 从文件中读取列表
    #new_list = read_list_from_file_3d('my_file.txt')
    l1_list = read_list_from_file_1d('labels_color/l1_class_name.txt')
    l2_list = read_list_from_file_2d('labels_color/l2_class_name.txt')
    l3_list = read_list_from_file_3d('labels_color/l3_class_name.txt')

    # 打印读取的列表
    #print("l1_list:",l1_list)
    #print("l2_list:", l2_list)
    #print("l3_list:", l3_list)

    directory = "./datasets/多级分类"  # 替换为要遍历的目录路径

    for root, dirs, files in os.walk(directory):
        for dir in dirs:
            if " " in dir:
                new_dir = dir.replace(' - 副本', "")
                os.rename(os.path.join(root, dir), os.path.join(root, new_dir))