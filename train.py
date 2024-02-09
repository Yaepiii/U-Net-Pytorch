# author: baiCai
from network_files.u_net import U_net
from utils import transforms as T
from My_Dataset import My_Dataset
from utils import Loss
import matplotlib.pyplot as plt
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim

# 训练的预处理方法获取
class SegmentationPresetTrain:
    def __init__(self, base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        # 初始化
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)

        # 随机裁剪
        trans = [T.RandomResize(min_size, max_size)]
        # 随机翻转
        if hflip_prob > 0:
            trans.append(T.RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            trans.append(T.RandomVerticalFlip(vflip_prob))
        trans.extend([  # 用于在列表末尾一次性追加另一个序列中的多个值
            T.RandomCrop(crop_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)

# 获取预处理方法
def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    base_size = 565
    crop_size = 480

    if train:
        # 训练模式获取的预处理方法
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return T.Compose([
            T.RandomCrop(480),
            T.ToTensor(),
            T.Normalize(mean, std)]
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size', type=int, default=2, help='input batch size')
    parser.add_argument(
        '--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    opt = parser.parse_args()

    # 设置基本参数信息
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 如果GPU不够用，可以用cpu
    # device = torch.device('cpu')
    batch_size= opt.batch_size
    epoch = opt.nepoch
    num_classes = 2 # 1(object)+1(background)
    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None
    # 加载数据
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    train_dataset = My_Dataset(opt.dataset,train=True,transforms=get_transform(train=True, mean=mean, std=std))
    val_dataset = My_Dataset(opt.dataset,train=False,transforms=get_transform(train=False, mean=mean, std=std))
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=True)
    # 创建模型
    model = U_net(3,num_classes,bilinear=True) # 输入通道数，输出通道数
    model.to(device)
    # 定义优化器
    params = [p for p in model.parameters() if p.requires_grad] # 定义需要优化的参数
    sgd = optim.SGD(params,lr=0.01,momentum=0.99,weight_decay=1e-4)  # weight_decay权重衰减系数(正则化项之前的系数),防止过拟合
    scheduler = optim.lr_scheduler.StepLR(sgd, step_size=5, gamma=0.5)
    # 开始训练
    model.train()
    losses = []
    for e in range(epoch):
        scheduler.step()
        loss_temp = 0
        for i,(image,mask) in enumerate(train_loader):
            image,mask = image.to(device),mask.to(device)
            output = model(image)
            # ‘out':out(4,2,480,480)
            loss = Loss.criterion(output, mask, loss_weight, num_classes=num_classes, ignore_index=255)
            loss_temp += loss.item()
            sgd.zero_grad()
            loss.backward()
            sgd.step()
        losses.append(loss_temp/(i+1))
        print(f'第{e+1}个epoch,平均损失loss={loss_temp/(i+1)}')
    # 保存权重
    name = 'save_weights/u_net.pth'
    torch.save(model.state_dict(),name)

    fig = plt.figure()
    plt.plot(range(epoch), losses)
    plt.xlabel('epoch'), plt.ylabel('loss')
    plt.show()

    with torch.no_grad():
        model.eval()
        loss_temp = 0
        for i,(image,mask) in enumerate(val_loader):
            image,mask = image.to(device),mask.to(device)
            output = model(image)
            loss = Loss.criterion(output, mask, loss_weight, num_classes=num_classes, ignore_index=255)
            loss_temp += loss.item()
        print(f'预测的平均损失loss={loss_temp / (i + 1)}')



if __name__ == '__main__':
    main()