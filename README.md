# U-Net-Pytorch
This repo is implementation for U-Net(https://arxiv.org/abs/1505.04597) in pytorch. The model is in `network_files/u_net.py`.

It is tested with pytorch-2.0.

# Download data and running

```
git clone https://github.com/Yaepiii/U-Net-Pytorch
cd U-Net-Pytorch
pip install torch
```

or, you are in anaconda:

```
conda create -n U-Net-Pytorch python=3.8
conda activate U-Net-Pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

Download dataset DRIVE(https://paperswithcode.com/dataset/drive), and the structure of the dataset:

```
├─test
│  ├─1st_manual
│  ├─2nd_manual
│  ├─images
│  └─mask
└─training
    ├─1st_manual
    ├─images
    └─mask
```

Training

```
python train.py --dataset=<your dataset path> --nepoch=50 --batch_size=2
```

Performance

<div align=center>
<img src="https://github.com/Yaepiii/U-Net-Pytorch/assets/75295024/4c894d0e-e38c-4f40-a184-b57aaffe976a" width="480" height="450">
</div>

Test result
```
The mean loss=0.41284542381763456
```

Predict
```
python predict.py
```

<div align=center>
<img src="https://github.com/Yaepiii/U-Net-Pytorch/assets/75295024/763c0d4f-0b53-4bdb-a1c0-461a3d0e6e03" width="360" height="360"><img src="https://github.com/Yaepiii/U-Net-Pytorch/assets/75295024/00fbdc2a-a474-4d40-a04d-c904104a6da6" width="360" height="360">
</div>










