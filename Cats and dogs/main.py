import argparse
import copy
import os
import torch.nn as nn
import torch
import torch.optim as optim
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from customdataset import MycustomDataset
from timm.loss import LabelSmoothingCrossEntropy
from adamp import AdamP
from utils import train, evaluate

def main(opt) :

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Augmentation 적용
    train_transform = A.Compose([
        A.Resize(width=224,height=224),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.7),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomShadow(p=0.5),
        A.RandomFog(p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(width=224, height=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    # 데이터셋 및 데이터로더 생성
    train_dataset = MycustomDataset(img_path=opt.train_path,transform=train_transform)
    val_dataset = MycustomDataset(img_path=opt.val_path, transform=val_transform)
    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    # 모델 생성 (ResNet-50)
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 클래스 개수에 맞게 마지막 레이어 수정
    model.to(device)

    # 손실 함수, 옵티마이저 설정
    criterion = LabelSmoothingCrossEntropy()
    criterion = criterion.to(device)
    # optimizer
    optimizer = AdamP(net.parameters(), lr=opt.lr)

    # scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # model.pt save dir
    save_dir = opt.save_path
    os.makedirs(save_dir, exist_ok=True)
    # train(num_epoch, model, train_loader, val_loader, criterion,
    #           optimizer, scheduler, save_dir, device)
    train(opt.epoch, net, train_loader, val_loader, criterion, optimizer,
          scheduler, save_dir, device)
    evaluate(net, val_loader, criterion, device)

def parse_opt() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", type=str, default="./dataset/train",
                        help="train data path")
    parser.add_argument("--val-path" ,type=str, default="./dataset/valid",
                        help="val data path")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="batch size")
    parser.add_argument("--epoch", type=int, default=100,
                        help="epoch number")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="lr number")
    parser.add_argument("--save-path", type=str, default="./weights",
                        help="save mode path")
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
