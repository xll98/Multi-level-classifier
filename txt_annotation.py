# ------------------------------------------------#
#   进行训练前需要利用这个文件生成cls_train.txt
# ------------------------------------------------#
import os

if __name__ == "__main__":
    # ---------------------#
    #   训练集所在的路径
    # ---------------------#
    # datasets_path   = "datasets"
    # datasets_path = "/home/share/DeeplearningData/Paimian/img_whh/"
    # datasets_path = "/home/share/DeeplearningData/Paimian/SKU_DOWN/"
    datasets_path = "G:/DeeplLearningPoject/PaimianClass/datasets"

    l1_path = os.listdir(datasets_path)
    l1_names = sorted(l1_path)

    list_file = open('cls_train.txt', 'w', encoding='utf-8')
    for l1_cls_id, l1_type_name in enumerate(l1_names):
        l2_path = os.path.join(datasets_path, l1_type_name)
        if not os.path.isdir(l2_path):
            continue
        l2_list = os.listdir(l2_path)
        l2_names = sorted(l2_list)
        for l2_cls_id, l2_type_name in enumerate(l2_names):
            l3_path = os.path.join(l2_path, l2_type_name)
            if not os.path.isdir(l3_path):
                continue
            l3_list = os.listdir(l3_path)
            l3_names = sorted(l3_list)
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
    list_file.close()
    print("finished!")
