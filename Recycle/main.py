import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.utils.data import DataLoader
from customdata import customData
from timm.loss import LabelSmoothingCrossEntropy
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform
train_transform = A.Compose([
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.RandomFog(),
    A.RandomSnow(),
    A.RandomShadow(),
    A.RandomRotate90(),
    A.RandomBrightnessContrast(),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
val_transform = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# dataset
train_dataset = customData('./dataset/train/', train_transform)
val_dataset = customData('./dataset/val/', val_transform)
test_dataset = customData('./dataset/test', val_transform)

# data loader
train_loader = DataLoader(train_dataset, batch_size=120, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=120, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# model call
model = models.resnet18(pretrained=True)
# print(model) (fc): Linear(in_features=512, out_features=1000, bias=True)
model.fc = nn.Linear(in_features=512, out_features=12)  # out_features 수정

model.to(device)

# HyParameter
criterion = LabelSmoothingCrossEntropy()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)


# train
def train(num_epoch, model, train_loader, val_loader, criterion, optimizer, device):
    total = 0
    best_loss = 9999

    for epoch in range(num_epoch):
        for idx, (images, labels) in enumerate(tqdm(train_loader, desc="train")):
            image, label = images.to(device), labels.to(device)

            output = model(image)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, argmax = torch.max(output, 1)
            acc = (labels == argmax).float().mean()
            total += labels.size(0)

            if (idx + 1) % 10 == 0:
                print('Epoch [{}/{}] Step [{}/{}] Loss {:.4f} Acc {:.2f}'.format(
                    epoch + 1, num_epoch, idx + 1, len(train_loader), loss.item(), acc.item() * 100
                ))
        avrg_loss, val_acc = validation(epoch, model, val_loader, criterion, device)

        if avrg_loss < best_loss:
            best_loss = avrg_loss
            torch.save(model.state_dict(), './best.pt')

    torch.save(model.state_dict(), './last.pt')


# val
def validation(epoch, model, val_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        total_loss = 0
        cnt = 0
        batch_loss = 0
        for idx, (images, labels) in enumerate(tqdm(val_loader, desc="validation")):
            image, label = images.to(device), labels.to(device)

            output = model(image)
            loss = criterion(output, label)
            batch_loss += loss.item()

            total += image.size(0)
            _, argmax = torch.max(output, 1)
            correct += (label == argmax).sum().item()
            total_loss += loss
            cnt += 1

    avrg_loss = total_loss / cnt
    val_acc = (correct / total * 100)
    print('Acc : {:.2f}% Loss : {:.4f}'.format(
        val_acc, avrg_loss
    ))
    model.train()
    return avrg_loss, val_acc


# test
def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for idx, (images, labels) in enumerate(tqdm(test_loader, desc="test")):
            image, label = images.to(device), labels.to(device)
            output = model(image)
            _, argmax = torch.max(output, 1)

            total += image.size(0)
            correct += (label == argmax).sum().item()

        acc = correct / total * 100
        print('acc for {} image : {:.2f}%'.format(
            total, acc
        ))


if __name__ == '__main__':
    train(num_epoch=10, model=model, train_loader=train_loader, val_loader=val_loader, criterion=criterion,
          optimizer=optimizer, device=device)

    model.load_state_dict(torch.load('./best.pt', map_location=device))
    test(model, test_loader, device)
