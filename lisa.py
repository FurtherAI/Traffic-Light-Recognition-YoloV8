import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation

class LISA(Dataset):
    def __init__(self, split='train'):
        self.split = 'train' if split == 'train' else 'val'
        if split == 'train':
            with open('lisa/train_images.txt', 'r') as images:
                self.image_paths = [pth.strip() for pth in images.readlines()]

            self.transforms = A.Compose([
                # A.RandomSizedBBoxSafeCrop(width=416, height=416, p=1.0),
                A.Resize(416, 416),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.CLAHE(p=0.4),
                # A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, always_apply=True),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), always_apply=True),
                ToTensorV2(transpose_mask=False, always_apply=True)
            ], bbox_params=A.BboxParams(format='yolo'))
        else:
            with open('lisa/val_images.txt', 'r') as images:
                self.image_paths = [pth.strip() for pth in images.readlines()]

            self.transforms = A.Compose([
                A.Resize(416, 416),
                # A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, always_apply=True),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), always_apply=True),
                ToTensorV2(transpose_mask=False, always_apply=True)
            ], bbox_params=A.BboxParams(format='yolo'))
        
        self.image_paths = np.sort(np.array(self.image_paths, dtype=np.bytes_)) # [0:32]
            
    def collate_fn(self, batch):
        images = torch.stack([data[0] for data in batch], axis=0)

        labels = []
        for idx, (img, bboxes, label) in enumerate(batch):
            labels.append(torch.cat((torch.full((bboxes.shape[0], 1), idx), label, bboxes), dim=1))
        
        labels = torch.cat(labels, dim=0)
        return images, labels, None, None

    # def collate_fn(batch):
    #     """YOLOv8 collate function, outputs dict."""
    #     im, label = zip(*batch)  # transposed
    #     for i, lb in enumerate(label):
    #         lb[:, 0] = i  # add target image index for build_targets()
    #     batch_idx, cls, bboxes = torch.cat(label, 0).split((1, 1, 4), dim=1)
    #     return {
    #         'img': torch.stack(im, 0),
    #         'cls': cls,
    #         'bboxes': bboxes,
    #         'batch_idx': batch_idx.view(-1)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # opencv - HWC, pytorch - CHW, albumentations - HWC
        pth = self.image_paths[idx].decode('UTF-8')[:-4]
        # flip the channel order because it was saved from opencv which is BGR, not RGB
        img = np.flip(np.ascontiguousarray(np.load(pth + '.npy').transpose(1, 2, 0)), axis=-1)  # HWC
        labels = []
        with open(f'{self.split}/labels/{idx}.txt') as annot:
            labels = [line.strip().split(' ') for line in annot.readlines()]
            labels = [[float(coord) for coord in label[1:]] + [int(label[0])] for label in labels] # xywh normalized
            labels = torch.tensor(labels).view(-1, 5)
            labels[:, 2:4].clamp_(max=torch.min(2 * (labels[:, :2]), 2 * (1 - labels[:, :2])))

        out = self.transforms(image=img, bboxes=labels)
        img = out['image']  # CHW - pytorch format
        labels = out['bboxes']
        if len(labels) != 0:
            bboxes = torch.stack([torch.tensor(bbox[:4], dtype=torch.float32) for bbox in labels], dim=0)
            labels = torch.tensor([bbox[4].item() for bbox in labels], dtype=torch.int32).unsqueeze(1)
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0, 1), dtype=torch.int32)

        return img, bboxes, labels

if __name__ == '__main__':
    x = LISA(split='val')
