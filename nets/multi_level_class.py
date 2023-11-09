import torch.nn as nn
import torch
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchsummary import summary

from nets.mobilenet import MobileNetV1
from nets.inception_resnet_v2 import InceptionResNetV2
from utils.utils import get_num_list
from torch.nn.utils.rnn import pad_sequence


# from nets.mobilenet import MobileNetV1
# from nets.inception_resnet_v2 import InceptionResNetV2


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class inception_resnetV2(nn.Module):
    def __init__(self):
        super(inception_resnetV2, self).__init__()
        self.model = InceptionResNetV2(5, 10, 5)

    def forward(self, x):
        x = self.model.stem(x)
        x = self.model.inception_resnet_a(x)
        x = self.model.reduction_a(x)
        x = self.model.inception_resnet_b(x)
        x = self.model.reduction_b(x)
        x = self.model.inception_resnet_c(x)
        return x, None


class mobilenet(nn.Module):
    def __init__(self):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1()
        del self.model.fc
        del self.model.avg

    def forward(self, x):
        y = self.model.stage1_part(x)
        y = self.model.stage2_part(y)
        y = self.model.stage3_part(y)
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x, y


class MultiLevelClassifier(nn.Module):
    def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=256, level1_num_classes=None,
                 level2_num_classes_list=None, level3_num_classes_list=None):
        super(MultiLevelClassifier, self).__init__()
        self.backoneName = backbone
        if backbone == "mobilenet":
            self.backbone = mobilenet()
            flat_shape = 1024
        elif backbone == "inception_resnetv2":
            self.backbone = inception_resnetV2()
            flat_shape = 2048
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv2.'.format(backbone))
        if len(level2_num_classes_list) != level1_num_classes:
            raise ValueError(
                '第一层的大小与 第二层的输入size 不一致 level1_num_classes - `{}` ,level2_size - {}'.format(
                    level1_num_classes, len(level2_num_classes_list)))
        for i, level3_num_classifier in enumerate(level3_num_classes_list):
            if len(level3_num_classifier) != level2_num_classes_list[i]:
                raise ValueError(
                    '第二层的第 -`{}` 个大小与 第三层的输入size 不一致 level2_num_classes - `{}`,level3_size - {}'.format(
                        i, level2_num_classes_list[i], len(level3_num_classifier)))
        # 将二维的level3_num_classes_list展开成一维的list
        self.level2_num_classes_list = level2_num_classes_list
        level3_num_classes_list_1d = [item for sublist in level3_num_classes_list for item in sublist]

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        self.level1_classifier = nn.Sequential(
            nn.Linear(flat_shape, embedding_size, bias=False),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, level1_num_classes)
        )

        self.level2_classifier = nn.ModuleList([
            nn.Sequential(
                nn.Linear(flat_shape, embedding_size, bias=False),
                nn.LayerNorm(embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, level2_classes)
            ) for level2_classes in level2_num_classes_list
        ])

        # 定义一个线性层用于学习注意力权重
        self.level2_attention_weights = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * flat_shape, 1),
            ) for level2_classes in level2_num_classes_list
        ])

        self.level3_classifiers = nn.ModuleList()
        for level3_classes in level3_num_classes_list:
            level3_classifier = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(flat_shape, embedding_size, bias=False),
                    nn.LayerNorm(embedding_size),
                    nn.ReLU(),
                    nn.Linear(embedding_size, level3_class)
                ) for level3_class in level3_classes
            ])
            self.level3_classifiers.append(level3_classifier)

    def forward(self, x):
        features_x, features_y = self.backbone(x)
        # features_x = self.backbone(x)
        features_x = self.avg(features_x)
        features_x = features_x.view(features_x.size(0), -1)
        features_x = self.Dropout(features_x)

        features_y = self.avg(features_y)
        features_y = features_y.view(features_y.size(0), -1)
        features_y = self.Dropout(features_y)

        features = torch.stack((features_x, features_y), dim=1)

        level1_features, level2_features, level3_features = [], [], []
        for feature_xy in features:
            # for feature in features:
            # feature = feature.view(1, -1)
            feature, feature_y = feature_xy
            use_level1_classifier = self.level1_classifier
            level1_feature = use_level1_classifier(feature)

            # 使用softmax和argmax选择level2 类别分类器
            selected_level1_class_label = torch.argmax(F.softmax(level1_feature, dim=-1), dim=-1)
            # 根据选择的小类别标签获取相应的小类别分类器
            use_level2_classifier = self.level2_classifier[selected_level1_class_label]
            feature_tmp = feature * 0.6 + feature_y * 0.4
            # feature_tmp = feature
            level2_feature = use_level2_classifier(feature_tmp)

            # 3级分类器的标签
            selected_level2_class_label = torch.argmax(F.softmax(level2_feature, dim=-1), dim=-1)

            # 计算在3级分类器中，使用哪一个分类器
            use_level3_classifier = self.level3_classifiers[selected_level1_class_label][selected_level2_class_label]
            level3_feature = use_level3_classifier(feature)

            level1_features.append(level1_feature)
            level2_features.append(level2_feature)
            level3_features.append(level3_feature)

        level1_features = pad_sequence(level1_features, batch_first=True, padding_value=-1)
        level2_features = pad_sequence(level2_features, batch_first=True, padding_value=-1)
        level3_features = pad_sequence(level3_features, batch_first=True, padding_value=-1)

        #level1_features = torch.stack(level1_features)
        #level2_features = torch.stack(level2_features)
        #level3_features = torch.stack(level3_features)

        return level1_features, level2_features, level3_features


if __name__ == "__main__":
    # level1_classes = 5  # 大类别的数量
    # level2_classes_list = [3, 3, 4, 3, 3]  # 不同大类别中的中类别数量列表
    # level3_classes_list = [[2, 1, 1], [2, 2, 2], [1, 1, 1, 1], [1, 1, 1], [2, 2, 2]]  # 不同中类别中的小类别数量列表

    # level1_classes = 10  # 大类别的数量
    # level2_classes_list = [3, 3, 4, 3, 3, 3 ,3 ,3, 3, 3]  # 不同大类别中的中类别数量列表
    # level3_classes_list = [[2,1,1], [2,2,2], [1,1,1,1], [1,1,1], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2], [2,2,2]]  # 不同中类别中的小类别数量列表

    annotation_path = "../cls_train.txt"
    # --------------------------------------------------------#
    #   输入图像大小，常用设置如[112, 112, 3]
    # --------------------------------------------------------#
    input_shape = [112, 112, 3]
    level1_classes, level2_classes_list, level3_classes_list = get_num_list(annotation_path)

    print("level1_classes", level1_classes)
    print("level2_classes_list", level2_classes_list)
    print("level3_classes_list", level3_classes_list)

    net = MultiLevelClassifier(backbone="mobilenet", level1_num_classes=level1_classes,
                               level2_num_classes_list=level2_classes_list, level3_num_classes_list=level3_classes_list)
    net = net.cuda()
    net.eval()
    a = torch.randn(4, 3, 112, 112)
    out1, out2, out3 = net(a.cuda())

    list_of_tensors = [torch.tensor([1, 2, 3]), torch.tensor([4,5,6]), torch.tensor([5, 6, 7])]

    # 使用torch.cat函数将这些tensor在第一维度连接起来，并使用 -1 填充
    padded_tensors = pad_sequence(list_of_tensors, batch_first=True, padding_value=-1)
    print("padded_tensors",padded_tensors)
    stack_tesnor = torch.stack(list_of_tensors)
    print("stack_tesnor",stack_tesnor)

    # summary(net, (3, 112, 112))
    # file_path = 'model_weights.pth'

    # 使用 torch.save 函数保存模型的权重
    # torch.save(net.state_dict(), file_path)
    print(out1, out2, out3)
    #print(net)
