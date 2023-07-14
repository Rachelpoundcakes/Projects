import os
import glob
import torch
import torchvision
from torch.utils.data import Dataset
# from PIL import Image
import cv2


class customData(Dataset):
    def __init__(self, path, transform=None):
        self.all_image_path = glob.glob(os.path.join(path, '*', '*.png'))
        self.transform = transform
        label_name = os.listdir(path)
        self.label_dict = {}
        for idx, label in enumerate(label_name):
            self.label_dict[label] = int(idx)

    def __getitem__(self, item):
        image_path = self.all_image_path[item]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_name = image_path.split('\\')[1]  # 윈도우에서 수정
        label = self.label_dict[label_name]

        if self.transform is not None:
            image = self.transform(image=image)['image']

        return image, label

    def __len__(self):
        return len(self.all_image_path)

if __name__ == '__main__':
    test = customData('./dataset/train/', transform=None)
    for i in test:
        pass