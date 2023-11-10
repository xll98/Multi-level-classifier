import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from nets.multi_level_class import MultiLevelClassifier as clssnet
from utils.utils import preprocess_input, resize_image, show_config
from utils.utils import read_list_from_file_1d, read_list_from_file_2d, read_list_from_file_3d,get_num_list_fromlabel


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和backbone需要修改！
# --------------------------------------------#
class ClassifierNet(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测要修改model_path，指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表准确度较高，仅代表该权值在验证集上泛化性能较好。
        # --------------------------------------------------------------------------#
        "model_path": "model_data/ep002-loss0.020-val_loss0.018-acc0.996.pth",
        # --------------------------------------------------------------------------#
        #   输入图片的大小。
        # --------------------------------------------------------------------------#
        "input_shape": [112, 112, 3],
        # --------------------------------------------------------------------------#
        #   所使用到的主干特征提取网络
        # --------------------------------------------------------------------------#
        # "backbone"      : "mobilenet",
        "backbone": "mobilenet",
        # -------------------------------------------#
        #   是否进行不失真的resize
        # -------------------------------------------#
        "letterbox_image": True,
        # -------------------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------------------#
        "cuda": False,
        # ------------------------------------------#
        # 标签文件路径
        # ------------------------------------------#
        "label_path": "labels"
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化Facenet
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.l3_list = None
        self.l2_list = None
        self.l1_list = None
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.generate()

        show_config(**self._defaults)

    def generate(self):
        # ---------------------------------------------------#
        #   载入模型与权值
        # ---------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        l1_label_path = self.label_path + '/l1_class_name.txt'
        l2_label_path = self.label_path + '/l2_class_name.txt'
        l3_label_path = self.label_path + '/l3_class_name.txt'

        self.l1_list = read_list_from_file_1d(l1_label_path)
        self.l2_list = read_list_from_file_2d(l2_label_path)
        self.l3_list = read_list_from_file_3d(l3_label_path)
        level1_classes, level2_classes_list, level3_classes_list = get_num_list_fromlabel(l1_label_path, l2_label_path,
                                                                                          l3_label_path)
        self.net = clssnet(backbone=self.backbone,level1_num_classes=level1_classes,
                                 level2_num_classes_list=level2_classes_list,
                                 level3_num_classes_list=level3_classes_list).eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    # ---------------------------------------------------#
    #   检测图片
    # ---------------------------------------------------#
    def detect_image(self, image_1):
        # ---------------------------------------------------#
        #   图片预处理，归一化
        # ---------------------------------------------------#
        with torch.no_grad():
            image_1 = resize_image(image_1, [self.input_shape[1], self.input_shape[0]],
                                   letterbox_image=self.letterbox_image)

            photo_1 = torch.from_numpy(
                np.expand_dims(np.transpose(preprocess_input(np.array(image_1, np.float32)), (2, 0, 1)), 0))

            if self.cuda:
                photo_1 = photo_1.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            cls_1, cls_2, cls_3 = self.net(photo_1)
            prob1 = F.softmax(cls_1, dim=1)
            prob2 = F.softmax(cls_2, dim=1)
            prob3 = F.softmax(cls_3, dim=1)
            clsid_l1 = torch.argmax(prob1, dim=-1)
            clsid_l2 = torch.argmax(prob2, dim=-1)
            clsid_l3 = torch.argmax(prob3, dim=-1)
            probability_l1 = torch.max(prob1, dim=-1)
            probability_l2 = torch.max(prob2, dim=-1)
            probability_l3 = torch.max(prob3, dim=-1)
            cls_name1 = self.l1_list[clsid_l1]
            cls_name2 = self.l2_list[clsid_l1][clsid_l2]
            cls_name3 = self.l3_list[clsid_l1][clsid_l2][clsid_l3]

            print("cls_name1:%s probability_l1:" % cls_name1, probability_l1.values)
            print("cls_name2:%s probability_l2:" % cls_name2, probability_l2.values)
            print("cls_name3:%s probability_l3:" % cls_name3, probability_l3.values)

            # ---------------------------------------------------#
            #   计算二者之间的距离
            # ---------------------------------------------------#

        plt.subplot(1, 1, 1)
        plt.imshow(np.array(image_1))
        #plt.title('Class:%s Probability:%.2f,%.2f,%.2f' % (cls_name3, probability_l1, probability_l2, probability_l3))

        plt.show()
