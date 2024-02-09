# author: baiCai
import os
import time

import torch
from utils import transforms as  T
import numpy as np
from PIL import Image

from network_files.u_net import U_net

def main():
    # 设置基本参数,需要自己改变路径参数
    classes = 1
    weights_path = "./save_weights/u_net.pth"
    img_path = "../DRIVE/test/images/03_test.tif" # ，另外可以改变预测的图像对象
    roi_mask_path = "../DRIVE/test/mask/03_test_mask.gif"
    # 这个是图像归一化的参数
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    # 获取设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 创建模型，输入通道数为3，输出为2
    model = U_net(3,classes+1)
    # 加载权重
    model.load_state_dict(torch.load(weights_path))
    model.to(device)
    # 打开相关路径
    roi_img = Image.open(roi_mask_path).convert('L')
    original_img = Image.open(img_path).convert('RGB')
    # 预处理, T.RandomCrop(480)表示随机裁剪
    data_transform = T.Compose([T.RandomCrop(480),
                                         T.ToTensor(),
                                         T.Normalize(mean=mean, std=std)])
    img,roi_img = data_transform(original_img,roi_img)
    # 将三维的数据转为四维，因为需要添加batch这个维度
    img = torch.unsqueeze(img, dim=0)
    # 将roi转为array，方便后期处理
    roi_img = np.array(roi_img)
    model.eval()  # 进入验证模式
    with torch.no_grad():
        # 初始化一个全黑的图像，后期将白色的添加进去，就可以得到预测的mask值
        # img_height, img_width = img.shape[-2:]
        # init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        # model(init_img)
        output = model(img.to(device))
        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不敢兴趣的区域像素设置成0(黑色)
        prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("test_result.png")

if __name__ == '__main__':
    main()
