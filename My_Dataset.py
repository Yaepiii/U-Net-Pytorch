# author: baiCai

import os
import torch
from torch import nn
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class My_Dataset(Dataset):
    def __init__(self,root,train=True,transforms=None):
        '''
        :param root: 路径，比如.\data
        :param train: 加载的数据集为训练还是测试集，默认为训练集
        :param transforms:  预处理方法
        '''
        super(My_Dataset, self).__init__()
        # 获取基础路径
        self.flag = 'training' if train else 'test'
        self.root = os.path.join(root,'DRIVE',self.flag) # ..\DRIVE\training
        # 获取所需的路径
        self.img_name = [i for i in os.listdir(os.path.join(self.root,'images'))] # '21_training.tif'
        self.img_list = [os.path.join(self.root,'images',i) for i in self.img_name] # .\data\DRIVE\training\images\21_training.tif
        self.manual = [os.path.join(self.root,'1st_manual',i.split('_')[0])+'_manual1.gif' for i in self.img_name ] # .\data\DRIVE\training\1st_manual\21_manual1.gif
        self.roi_mask = [os.path.join(self.root,'mask',i.split('_')[0]+'_'+self.flag+'_mask.gif') for i in self.img_name] # .\data\DRIVE\training\mask\21_training_mask.gif
        # 初始化其它变量
        self.transforms = transforms

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # 打开图像(RGB模式)、manual(L-灰度图)、roi_mask（L-灰度图）
        image = Image.open(self.img_list[idx])
        manual = Image.open(self.manual[idx]).convert('L')
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        # 对图像进行处理
        # 将对象设置为1，背景设置为0，不感兴趣的区域设置为255
        manual = np.array(manual) / 255  # 白色（对象）为255，变为了1
        roi_mask = 255 - np.array(roi_mask) # 不感兴趣区域为黑色（0），变为了255 ； 感兴趣为白色（255），变为了0
        mask = np.clip(manual+roi_mask,a_min=0,a_max=255)   # 裁剪，将数组的值限制在[a_min, a_max]
        # 不感兴趣的为255，对象(前景)为1，背景为0
        mask = Image.fromarray(mask) # 转为PIL，方便做预处理操作
        if self.transforms is not None:
            image,mask = self.transforms(image,mask)
        return image,mask