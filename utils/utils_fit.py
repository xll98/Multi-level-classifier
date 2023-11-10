import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import evaluate

min_loss = 100.0

"""
    训练一个深度学习模型的一个epoch，并进行验证。

    参数:
    model_train (nn.Module): 训练模型
    model (nn.Module): 模型
    loss_history (LossHistory): 用于记录损失历史的对象
    optimizer (torch.optim.Optimizer): 优化器
    epoch (int): 当前epoch的索引
    epoch_step (int): 训练数据集的步数
    epoch_step_val (int): 验证数据集的步数
    gen (DataLoader): 训练数据生成器
    gen_val (DataLoader): 验证数据生成器
    Epoch (int): 总共的epoch数
    cuda (bool): 是否使用GPU加速
    fp16 (bool): 是否启用混合精度训练
    scaler (torch.cuda.amp.GradScaler): 用于混合精度训练的梯度缩放器
    save_period (int): 保存模型的周期
    save_dir (str): 模型保存的目录
    local_rank (int): 当前进程的本地排名
"""


def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
                  Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank):
    loss_l1_weight = 2  # 第1层loss函数的权重
    loss_l2_weight = 0.5  # 第2层loss函数的权重
    loss_l3_weight = 0.5  # 第3层loss函数的权重

    total_ce_loss_l1 = 0  # 第一层的总交叉熵损失
    total_ce_loss_l2 = 0  # 第二层的总交叉熵损失
    total_ce_loss_l3 = 0  # 第三层的总交叉熵损失
    total_loss = 0  # 总loss
    total_accuracy = 0  # 总准确率
    total_accuracy_l1 = 0  # 第1层总准确率
    total_accuracy_l2 = 0  # 第2层总准确率
    total_accuracy_l3 = 0  # 第3层总准确率

    val_total_ce_loss_l1 = 0  # 验证集第一层的总交叉熵损失
    val_total_ce_loss_l2 = 0  # 验证集第二层的总交叉熵损失
    val_total_ce_loss_l3 = 0  # 验证集第三层的总交叉熵损失
    val_total_loss = 0  # 验证集总loss
    val_total_accuracy = 0  # 验证集总准确率
    val_total_accuracy_l1 = 0  # 验证集第1层总准确率
    val_total_accuracy_l2 = 0  # 验证集第2层总准确率
    val_total_accuracy_l3 = 0  # 验证集第3层总准确率

    if local_rank == 0:
        print('Start Train')  # 打印开始训练的信息
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)  # 初始化进度条
    model_train.train()  # 设置模型为训练模式
    for iteration, batch in enumerate(gen):  # 遍历训练数据集
        if iteration >= epoch_step:
            break
        images, labels = batch  # 获取图像和标签
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                labels = labels.cuda(local_rank)

        optimizer.zero_grad()  # 梯度清零

        labels_l1 = labels[:, 0]
        labels_l2 = labels[:, 1]
        labels_l3 = labels[:, 2]

        if not fp16:
            cls_l1, cls_l2, cls_l3 = model_train(images)  # 前向传播
            _ce_loss_l1 = 0
            _ce_loss_l2 = 0
            _ce_loss_l3 = 0
            for i in range(images.size(0)):  # 遍历每个样本
                label_l1 = labels_l1[i].item()
                label_l2 = labels_l2[i].item()
                label_l3 = labels_l3[i].item()
                pared_l1 = torch.argmax(F.softmax(cls_l1[i], dim=-1), dim=-1).item()
                pared_l2 = torch.argmax(F.softmax(cls_l2[i], dim=-1), dim=-1).item()
                tmp_loss_l1 = nn.NLLLoss()(F.log_softmax(cls_l1[i].unsqueeze(0), dim=-1), torch.tensor([label_l1]).to(cls_l1.device))
                if pared_l1 == label_l1:  # 判断l1是否是正确的
                    tmp_loss_l2 = nn.NLLLoss()(F.log_softmax(cls_l2[i].unsqueeze(0), dim=-1), torch.tensor([label_l2]).to(cls_l1.device))
                else:
                    tmp_loss_l2 = tmp_loss_l1  # 超出范围时沿用_ce_loss_l1
                if pared_l2 == label_l2:  # 判断是否超出范围
                    tmp_loss_l3 = nn.NLLLoss()(F.log_softmax(cls_l3[i].unsqueeze(0), dim=-1), torch.tensor([label_l3]).to(cls_l1.device))
                else:
                    tmp_loss_l3 = tmp_loss_l2  # 超出范围时沿用_ce_loss_l1
                _ce_loss_l1 += tmp_loss_l1
                _ce_loss_l2 += tmp_loss_l2
                _ce_loss_l3 += tmp_loss_l3
            _ce_loss_l1 = _ce_loss_l1 / images.size(0)
            _ce_loss_l2 = _ce_loss_l2 / images.size(0)
            _ce_loss_l3 = _ce_loss_l3 / images.size(0)
            _loss = (loss_l1_weight * _ce_loss_l1) + (loss_l2_weight * _ce_loss_l2) + (
                    loss_l3_weight * _ce_loss_l3)
            _loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
        else:
            from torch.cuda.amp import autocast
            with autocast():
                cls_l1, cls_l2, cls_l3 = model_train(images)  # 前向传播
                _ce_loss_l1 = 0
                _ce_loss_l2 = 0
                _ce_loss_l3 = 0
                for i in range(images.size(0)):  # 遍历每个样本
                    label_l1 = labels_l1[i].item()
                    label_l2 = labels_l2[i].item()
                    label_l3 = labels_l3[i].item()
                    pared_l1 = torch.argmax(F.softmax(cls_l1[i], dim=-1), dim=-1).item()
                    pared_l2 = torch.argmax(F.softmax(cls_l2[i], dim=-1), dim=-1).item()
                    tmp_loss_l1 = nn.NLLLoss()(F.log_softmax(cls_l1[i].unsqueeze(0), dim=-1), torch.tensor([label_l1]).to(cls_l1.device))
                    if pared_l1 == label_l1:  # 判断l1是否是正确的
                        tmp_loss_l2 = nn.NLLLoss()(F.log_softmax(cls_l2[i].unsqueeze(0), dim=-1),
                                                   torch.tensor([label_l2]))
                    else:
                        tmp_loss_l2 = tmp_loss_l1  # 超出范围时沿用_ce_loss_l1
                    if pared_l2 == label_l2:  # 判断是否超出范围
                        tmp_loss_l3 = nn.NLLLoss()(F.log_softmax(cls_l3[i].unsqueeze(0), dim=-1),
                                                   torch.tensor([label_l3]))
                    else:
                        tmp_loss_l3 = tmp_loss_l2  # 超出范围时沿用_ce_loss_l1
                    _ce_loss_l1 += tmp_loss_l1
                    _ce_loss_l2 += tmp_loss_l2
                    _ce_loss_l3 += tmp_loss_l3
                _ce_loss_l1 = _ce_loss_l1 / images.size(0)
                _ce_loss_l2 = _ce_loss_l2 / images.size(0)
                _ce_loss_l3 = _ce_loss_l3 / images.size(0)
                _loss = (loss_l1_weight * _ce_loss_l1) + (loss_l2_weight * _ce_loss_l2) + (
                        loss_l3_weight * _ce_loss_l3)  # 根据各层权重计算总loss

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(_loss).backward()  # 梯度缩放
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新 scaler
        with torch.no_grad():
            ret_l1 = torch.argmax(F.softmax(cls_l1, dim=-1), dim=-1)
            ret_l2 = torch.argmax(F.softmax(cls_l2, dim=-1), dim=-1)
            ret_l3 = torch.argmax(F.softmax(cls_l3, dim=-1), dim=-1)
            # 计算l1的准确率
            accuracy_l1 = torch.mean((ret_l1 == labels_l1).type(torch.FloatTensor))
            # 计算l2的准确率
            accuracy_l2 = torch.mean(((ret_l2 == labels_l2) & (ret_l1 == labels_l1)).type(torch.FloatTensor))
            # 计算l3的准确率
            accuracy_l3 = torch.mean(
                ((ret_l3 == labels_l3) & (ret_l2 == labels_l2) & (ret_l1 == labels_l1)).type(
                    torch.FloatTensor))
        # 分别累加损失
        total_ce_loss_l1 += _ce_loss_l1.item()
        total_ce_loss_l2 += _ce_loss_l2.item()
        total_ce_loss_l3 += _ce_loss_l3.item()

        #  累加准确率
        total_accuracy_l1 += accuracy_l1.item()
        total_accuracy_l2 += accuracy_l2.item()
        total_accuracy_l3 += accuracy_l3.item()
        total_accuracy = total_accuracy_l3

        if local_rank == 0:
            pbar.set_postfix(**{'total_ce_loss_l1': total_ce_loss_l1 / (iteration + 1),
                                'total_ce_loss_l2': total_ce_loss_l2 / (iteration + 1),
                                'total_ce_loss_l3': total_ce_loss_l3 / (iteration + 1),
                                'total_accuracy': total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})  # 更新进度条显示的信息
            pbar.update(1)  # 更新进度条的步数
            cur_accuracy = (total_accuracy / (iteration + 1))
    if local_rank == 0:
        pbar.close()  # 关闭进度条
        print('Finish Train')  # 打印训练完成的信息
        print('Start Validation')  # 打印开始验证的信息
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)  # 初始化进度条
    model_train.eval()  # 设置模型为验证模式
    for iteration, batch in enumerate(gen_val):  # 遍历验证集
        if iteration >= epoch_step:
            break
        images, labels = batch  # 获取图像和标签
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                labels = labels.cuda(local_rank)

            optimizer.zero_grad()  # 梯度清零
            labels_l1 = labels[:, 0]
            labels_l2 = labels[:, 1]
            labels_l3 = labels[:, 2]
            cls_l1, cls_l2, cls_l3 = model_train(images)  # 前向传播
            _ce_loss_l1 = 0
            _ce_loss_l2 = 0
            _ce_loss_l3 = 0
            for i in range(images.size(0)):  # 遍历每个样本
                label_l1 = labels_l1[i].item()
                label_l2 = labels_l2[i].item()
                label_l3 = labels_l3[i].item()
                pared_l1 = torch.argmax(F.softmax(cls_l1[i], dim=-1), dim=-1).item()
                pared_l2 = torch.argmax(F.softmax(cls_l2[i], dim=-1), dim=-1).item()
                tmp_loss_l1 = nn.NLLLoss()(F.log_softmax(cls_l1[i].unsqueeze(0), dim=-1), torch.tensor([label_l1]).to(cls_l1.device))
                if pared_l1 == label_l1:  # 判断l1是否是正确的
                    tmp_loss_l2 = nn.NLLLoss()(F.log_softmax(cls_l2[i].unsqueeze(0), dim=-1), torch.tensor([label_l2]).to(cls_l1.device))
                else:
                    tmp_loss_l2 = tmp_loss_l1  # 超出范围时沿用_ce_loss_l1
                if pared_l2 == label_l2:  # 判断是否超出范围
                    tmp_loss_l3 = nn.NLLLoss()(F.log_softmax(cls_l3[i].unsqueeze(0), dim=-1), torch.tensor([label_l3]).to(cls_l1.device))
                else:
                    tmp_loss_l3 = tmp_loss_l2  # 超出范围时沿用_ce_loss_l1
                _ce_loss_l1 += tmp_loss_l1
                _ce_loss_l2 += tmp_loss_l2
                _ce_loss_l3 += tmp_loss_l3
            _ce_loss_l1 = _ce_loss_l1 / images.size(0)
            _ce_loss_l2 = _ce_loss_l2 / images.size(0)
            _ce_loss_l3 = _ce_loss_l3 / images.size(0)
            _loss = (loss_l1_weight * _ce_loss_l1) + (loss_l2_weight * _ce_loss_l2) + (
                    loss_l3_weight * _ce_loss_l3)

            ret_l1 = torch.argmax(F.softmax(cls_l1, dim=-1), dim=-1)
            ret_l2 = torch.argmax(F.softmax(cls_l2, dim=-1), dim=-1)
            ret_l3 = torch.argmax(F.softmax(cls_l3, dim=-1), dim=-1)
            # 计算l1的准确率
            accuracy_l1 = torch.mean((ret_l1 == labels_l1).type(torch.FloatTensor))
            # 计算l2的准确率
            accuracy_l2 = torch.mean(((ret_l2 == labels_l2) & (ret_l1 == labels_l1)).type(torch.FloatTensor))
            # 计算l3的准确率
            accuracy_l3 = torch.mean(
                ((ret_l3 == labels_l3) & (ret_l2 == labels_l2) & (ret_l1 == labels_l1)).type(
                    torch.FloatTensor))

            # 分别累加损失
            val_total_ce_loss_l1 += _ce_loss_l1.item()
            val_total_ce_loss_l2 += _ce_loss_l2.item()
            val_total_ce_loss_l3 += _ce_loss_l3.item()

            #  累加准确率
            val_total_accuracy_l1 += accuracy_l1.item()
            val_total_accuracy_l2 += accuracy_l2.item()
            val_total_accuracy_l3 += accuracy_l3.item()
            val_total_accuracy = val_total_accuracy_l3

        if local_rank == 0:
            pbar.set_postfix(**{'val_total_ce_loss_l1': val_total_ce_loss_l1 / (iteration + 1),
                                'val_total_ce_loss_l2': val_total_ce_loss_l2 / (iteration + 1),
                                'val_total_ce_loss_l3': val_total_ce_loss_l3 / (iteration + 1),
                                'val_total_accuracy': val_total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})  # 更新进度条显示的信息
            pbar.update(1)  # 更新进度条的步数
    if local_rank == 0:
        pbar.close()  # 关闭进度条
        print('Finish Validation')  # 打印验证完成的信息
        total_loss = (loss_l1_weight * total_ce_loss_l1) + (loss_l2_weight * total_ce_loss_l2) + (
                loss_l3_weight * total_ce_loss_l3)  # 根据各层权重计算总loss
        val_total_loss = (loss_l1_weight * val_total_ce_loss_l1) + (loss_l2_weight * val_total_ce_loss_l2) + (
                loss_l3_weight * val_total_ce_loss_l3)  # 根据各层权重计算总loss
        loss_history.append_loss(epoch, total_accuracy / epoch_step,
                                 total_loss / epoch_step,
                                 val_total_loss / epoch_step_val)  # 存储损失信息
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))  # 打印当前epoch和总epoch
        print('Total Loss: %.4f' % (total_loss / epoch_step))  # 打印总损失

        global min_loss
        cur_loss = total_loss / epoch_step  # 当前epoch的总损失
        cur_save = False  # 当前epoch是否需要保存模型
        if cur_loss <= min_loss:
            cur_save = True
            min_loss = cur_loss
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch or cur_save == True:
            torch.save(model.state_dict(),
                       os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f-acc%.3f.pth' % ((epoch + 1),
                                                                                            total_loss / epoch_step,
                                                                                            val_total_loss / epoch_step_val,
                                                                                            cur_accuracy)))  # 保存模型
