# author: baiCai

import torch
from torch import nn
from torch.nn import functional as F

# 上采样+拼接
class Up(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True):
        '''
        :param in_channels: 输入通道数
        :param out_channels:  输出通道数
        :param bilinear: 是否采用双线性插值，默认采用
        '''
        super(Up, self).__init__()
        if bilinear:
            # 双线性差值
            self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True) # 不改变通道数 512
            self.conv = doubleConv(in_channels,out_channels,in_channels//2) # 拼接后为1024，经历第一个卷积后512
        else:
            # 转置卷积实现上采样
            # 输出通道数减半，宽高增加一倍
            self.up = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2) # 1024,512
            self.conv = doubleConv(in_channels,out_channels)    # 1024,512

    def forward(self,x1,x2):
        # 上采样
        x1 = self.up(x1)    # 上采样:512;转置卷积:1024,512
        # 拼接
        x = torch.cat([x1,x2],dim=1) # 如果双线性插值,拼接后为1024;如果转置卷积,卷积后为512
        # 经历双卷积
        x = self.conv(x)    # 如果双线性插值,卷积后为256;转置卷积后为512
        return x

# 双卷积层
# noinspection LanguageDetectionInspection
def doubleConv(in_channels,out_channels,mid_channels=None):
    '''
    :param in_channels: 输入通道数
    :param out_channels: 双卷积后输出的通道数
    :param mid_channels: 中间的通道数，这个主要针对的是最后一个下采样和上采样层
    :return:
    '''
    if mid_channels is None:
        mid_channels = out_channels
    layer = []
    layer.append(nn.Conv2d(in_channels,mid_channels,kernel_size=3,padding=1,bias=False))
    layer.append(nn.BatchNorm2d(mid_channels))
    # inplace = True ,会改变输入数据的值,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
    layer.append(nn.ReLU(inplace=True))
    layer.append(nn.Conv2d(mid_channels,out_channels,kernel_size=3,padding=1,bias=False))
    layer.append(nn.BatchNorm2d(out_channels))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        self.mid_channels = mid_channels
        if self.mid_channels is None:
            self.mid_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.net(x)
        return y



# 下采样
def down(in_channels,out_channels):
    # 池化 + 双卷积
    layer = []
    layer.append(nn.MaxPool2d(2,stride=2))
    layer.append(doubleConv(in_channels,out_channels))
    return nn.Sequential(*layer)

class Down(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Down, self).__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self,x):
        y = self.net(x)
        return y

# 整个网络架构
class U_net(nn.Module):
    def __init__(self,in_channels,out_channels,bilinear=True,base_channel=64):
        '''
        :param in_channels: 输入通道数，一般为3，即彩色图像
        :param out_channels: 输出通道数，即网络最后输出的通道数，一般为2，即进行2分类
        :param bilinear: 是否采用双线性插值来上采样，这里默认采取
        :param base_channel: 第一个卷积后的通道数，即64
        '''
        super(U_net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # 输入
        self.in_conv = doubleConv(self.in_channels,base_channel)
        # 下采样
        self.down1 = down(base_channel,base_channel*2) # 64,128
        self.down2 = down(base_channel*2,base_channel*4) # 128,256
        self.down3 = down(base_channel*4,base_channel*8) # 256,512
        # 最后一个下采样，通道数不翻倍（因为双线性差值，不会改变通道数的，为了可以简单拼接，就不改变通道数）
        # 当然，是否采取双线新差值，还是由我们自己决定
        factor = 2  if self.bilinear else 1
        self.down4 = down(base_channel*8,base_channel*16 // factor) # 512,512;如果用转置卷积,512,1024

        # self.ic = DoubleConv(self.in_channels,base_channel)
        # self.d1 = Down(base_channel,base_channel*2)
        # self.d2 = Down(base_channel*2,base_channel*4)
        # self.d3 = Down(base_channel*4,base_channel*8)
        # self.d4 = Down(base_channel*8,base_channel*16 // factor)

        # 上采样 + 拼接
        self.up1 = Up(base_channel*16 ,base_channel*8 // factor,self.bilinear)  # 1024,256(双线性插值) 1024,512(转置卷积)
        self.up2 = Up(base_channel*8 ,base_channel*4 // factor,self.bilinear)   # 512,256(转置卷积)
        self.up3 = Up(base_channel*4 ,base_channel*2 // factor,self.bilinear)   # 256,128(转置卷积)
        self.up4 = Up(base_channel*2 ,base_channel,self.bilinear)               # 128,64(转置卷积)
        # 输出
        self.out = nn.Conv2d(in_channels=base_channel,out_channels=self.out_channels,kernel_size=1)

    def forward(self,x):
        # x1 = self.ic(x)
        # x2 = self.d1(x1)
        # x3 = self.d2(x2)
        # x4 = self.d3(x3)
        # x5 = self.d4(x4) # (批尺寸,通道数,图像高,图像宽)
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4) # (批尺寸,通道数,图像高,图像宽)
        # 不要忘记拼接
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.out(x)

        return {'out':out}