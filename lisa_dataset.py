from typing import Any
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import os

MEAN = [0.25016354, 0.18997617, 0.18944333]
STD = [0.14122078, 0.12227465, 0.11524759]

IMAGENET_MEAN = 0.485, 0.456, 0.406  # RGB mean
IMAGENET_STD = 0.229, 0.224, 0.225  # RGB standard deviation

class LISA(Dataset):
    def __init__(self, split='train'):
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
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        else:
            with open('lisa/val_images.txt', 'r') as images:
                self.image_paths = [pth.strip() for pth in images.readlines()]

            self.transforms = A.Compose([
                A.Resize(416, 416),
                # A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, always_apply=True),
                A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), always_apply=True),
                ToTensorV2(transpose_mask=False, always_apply=True)
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        
        self.image_paths = np.array(self.image_paths, dtype=np.bytes_) # [0:32]
            
    def collate_fn(self, batch):
        images = torch.stack([data[0] for data in batch], axis=0)

        # assuming square input
        yolo_reduction_factor = 32
        small_grid_wh = images[0].shape[1] // yolo_reduction_factor
        large_grid_wh = small_grid_wh * 2

        bbox13 = torch.zeros((images.shape[0], small_grid_wh, small_grid_wh, 4), dtype=torch.float32, device='cpu')
        labels13 = torch.full((images.shape[0], small_grid_wh, small_grid_wh), -1, dtype=torch.long, device='cpu')
        bbox26 = torch.zeros((images.shape[0], large_grid_wh, large_grid_wh, 4), dtype=torch.float32, device='cpu')
        labels26 = torch.full((images.shape[0], large_grid_wh, large_grid_wh), -1, dtype=torch.long, device='cpu')

        for batch_idx, (_, bboxes, labels) in enumerate(batch):
            # bboxes (n, 4)
            # labels (n)
            for box_idx in range(bboxes.shape[0]):
                box = bboxes[box_idx]
                center = ((box[2:] + box[:2]) / 2)
                scenter = (center // yolo_reduction_factor).long()
                lcenter = (center // (yolo_reduction_factor >> 1)).long()

                bbox13[batch_idx, scenter[1], scenter[0], :] = box
                labels13[batch_idx,  scenter[1], scenter[0]] = labels[box_idx]
                bbox26[batch_idx, lcenter[1], lcenter[0], :] = box
                labels26[batch_idx, lcenter[1], lcenter[0]] = labels[box_idx]

        return images, bbox13, labels13, bbox26, labels26

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # opencv - HWC, pytorch - CHW, albumentations - HWC
        label_to_int = {'stop' : 0, 'warning' : 1, 'go' : 2}
        pth = self.image_paths[idx].decode('UTF-8')[:-4]
        '''IMPORTANT flip the channel order because it was saved from opencv which is BGR, not RGB'''
        img = np.flip(np.ascontiguousarray(np.load(pth + '.npy').transpose(1, 2, 0)), axis=-1)  # HWC
        labels = []
        if os.path.exists(pth + '.txt'):
            with open(pth + '.txt') as annot:
                labels = [line.strip().split(',')[1:] for line in annot.readlines()]
                # int(float(coord) * 0.65)
                labels = [[int(float(coord) * 0.65) for coord in label[1:]] + label[:1] for label in labels]  # place bbox label after coordinates (xmin, ymin, xmax, ymax)

        out = self.transforms(image=img, bboxes=labels)
        img = out['image']  # CHW - pytorch format
        labels = out['bboxes']
        if len(labels) != 0:
            bboxes = torch.stack([torch.tensor(bbox[:4]) for bbox in labels], dim=0)
            labels = torch.tensor([label_to_int[bbox[4]] for bbox in labels])
        else:
            bboxes = torch.zeros((0, 4))
            labels = torch.zeros((0, 1))
        return img, bboxes, labels 

    def imgs_to_numpy(self):  # transform all images to npy files
        with open('images.txt', 'r') as images:
            image_paths = [pth.strip() for pth in images.readlines()]
        for img_path in image_paths:
            img = cv2.imread(img_path)
            img = cv2.resize(img, dsize=None, fx=0.65, fy=0.65, interpolation=cv2.INTER_LINEAR)
            img = np.transpose(img, (2, 0, 1))
            np.save(img_path[:-4] + '.npy', img)

    def calc_mean_std(self):  # img mean and standard deviation
        with open('images.txt', 'r') as images:
            image_paths = [pth.strip() for pth in images.readlines()]
        indexes = np.random.randint(0, len(image_paths), size=(5000))
        means = []
        stds = []
        for idx in indexes:
            img = np.load(image_paths[idx][:-4] + '.npy') / 255
            means.append(img.reshape(3, -1).mean(axis=1))
            stds.append(img.reshape(3, -1).std(axis=1))
        means = np.stack(means, axis=0)
        stds = np.stack(stds, axis=0)
        print('channel mean: ', means.mean(axis=0))
        print('channel std: ', stds.std(axis=0))

    def calc_anchor(self):
        with open('images.txt', 'r') as images:
            image_paths = [pth.strip() for pth in images.readlines()]
        indexes = np.random.randint(0, len(image_paths), size=(5000))
        w, h, n = 0, 0, 0
        for idx in indexes:
            if not os.path.exists(image_paths[idx][:-4] + '.txt'): continue
            with open(image_paths[idx][:-4] + '.txt', 'r') as annot:
                # print(annot.readlines())
                boxes = [[int(coord) for coord in line.strip().split(',')[2:6]] for line in annot.readlines()]
                for bbox in boxes:
                    n += 1
                    w += bbox[2] - bbox[0]
                    h += bbox[3] - bbox[1]
        print('n:', n)
        print('width:', w / n)
        print('height:', h / n)


if __name__ == '__main__':
    x = LISA(split='val')
    img, bboxes, labels = x[1]
    img = img.to(dtype=torch.uint8)
    img_with_bbox = torchvision.utils.draw_bounding_boxes(img, bboxes, colors=(255, 255, 0))
    res = cv2.imwrite('out.jpg', np.flip(img_with_bbox.permute(1, 2, 0).numpy(), axis=-1))
    print(res)

    # pth = x.image_paths[0].decode('UTF-8')[:-4] + '.npy'
    # img = torch.from_numpy(np.load(pth))
    # img_with_bbox = torchvision.utils.draw_bounding_boxes(img, bboxes, colors=(255, 255, 0))
    # res = cv2.imwrite('out.jpg', img_with_bbox.permute(1, 2, 0).numpy())
    # print(res)
