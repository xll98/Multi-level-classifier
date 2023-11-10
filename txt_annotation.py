# ------------------------------------------------#
#   进行训练前需要利用这个文件生成cls_train.txt
# ------------------------------------------------#
import os

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

if __name__ == "__main__":
    # ---------------------#
    #   训练集所在的路径
    # ---------------------#
    # datasets_path   = "datasets"
    # datasets_path = "/home/share/DeeplearningData/Paimian/img_whh/"
    # datasets_path = "/home/share/DeeplearningData/Paimian/SKU_DOWN/"
    datasets_path = "G:/DeeplLearningPoject/PaimianClass/datasets/SkuMWHH"

    l1_path = os.listdir(datasets_path)
    l1_names = sorted(l1_path)
    list_level1 = []
    list_level2 = []
    list_level3 = []

    list_file = open('labels/cls_train.txt', 'w', encoding='utf-8')
    for l1_cls_id, l1_type_name in enumerate(l1_names):
        l2_path = os.path.join(datasets_path, l1_type_name)
        if not os.path.isdir(l2_path):
            continue
        l2_list = os.listdir(l2_path)
        l2_names = sorted(l2_list)
        tmp_list_l2 = []
        tmp_list_l3_2 = []
        for l2_cls_id, l2_type_name in enumerate(l2_names):
            l3_path = os.path.join(l2_path, l2_type_name)
            if not os.path.isdir(l3_path):
                continue
            l3_list = os.listdir(l3_path)
            l3_names = sorted(l3_list)
            tmp_list_l3_3 = []
            for l3_cls_id, l3_type_name in enumerate(l3_names):
                photos_path = os.path.join(l3_path, l3_type_name)
                if not os.path.isdir(photos_path):
                    continue
                photos_name = os.listdir(photos_path)

                for photo_name in photos_name:
                    name_src = '%s' % (os.path.join(os.path.abspath(l3_path), l3_type_name, photo_name))
                    # 去掉文件名中的空格
                    name_new = name_src.replace(' ', '')
                    os.rename(name_src, name_new)
                    list_file.write(
                        str(l1_cls_id) + ";" + str(l2_cls_id) + ";" + str(l3_cls_id) + ";" + name_new)
                    list_file.write('\n')
                tmp_list_l3_3.append(l3_type_name)
            tmp_list_l3_2.append(tmp_list_l3_3)
            tmp_list_l2.append(l2_type_name)
        list_level1.append(l1_type_name)
        list_level2.append(tmp_list_l2)
        list_level3.append(tmp_list_l3_2)

    write_list_to_file_1d(list_level1, "labels/l1_class_name.txt")
    write_list_to_file_2d(list_level2, "labels/l2_class_name.txt")
    write_list_to_file_3d(list_level3, "labels/l3_class_name.txt")
    list_file.close()
    print("finished!")
