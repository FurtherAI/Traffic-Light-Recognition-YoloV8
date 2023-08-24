import os
# from ultralytics import YOLO
from test_ultra import YOLO
import torch
import torch.nn as nn
import numpy as np
import torchvision
import cv2
from lisa import LISA
from kalman import KalmanFilterXYWH


cls_to_color = {0: 'red', 1: 'yellow', 2: 'green'}

model = YOLO('runs/detect/yolos_640/weights/best.pt')

class PersistentBoxes():
    kalman_filter = KalmanFilterXYWH()

    def __init__(self, persistence=2):
        self.boxes = np.empty((0, 8))
        self.cov = np.empty((0, 8, 8))
        self.cls = np.empty((0,))
        self.timer = np.empty((0,))

        self.persistence = persistence  # num updates to hold box
    
    def iou(self, boxes, boxes_, eps=1e-5):  # xyxy, last dim holding coordinates
        mins = boxes[..., :2]
        maxs = boxes[..., 2:]
        mins_ = boxes_[..., :2]
        maxs_ = boxes_[..., 2:]

        intersection = np.clip(np.minimum(maxs, maxs_) - np.maximum(mins, mins_), a_min=0, a_max=None)
        intersection = intersection[..., 0] * intersection[..., 1]  # intersection area

        parea, gtarea = (maxs - mins), (maxs_ - mins_)
        union = parea[..., 0] * parea[..., 1] + gtarea[..., 0] * gtarea[..., 1] - intersection + eps
        iou = intersection / union
        return iou
    
    def xyxy2xywh(self, boxes):
        wh = boxes[..., 2:4] - boxes[..., :2]
        xy = (boxes[..., 2:4] + boxes[..., :2]) / 2
        return np.concatenate([xy, wh], axis=-1)
    
    def xywh2xyxy(self, boxes):
        wh_ = boxes[..., 2:4] / 2
        xyxy = np.concatenate([np.clip(boxes[..., :2] - wh_, a_min=0, a_max=None), boxes[..., :2] + wh_], axis=-1)
        return xyxy

    def update(self, new_boxes, new_cls):
        new_boxes = new_boxes.numpy()
        new_cls = new_cls.numpy()
        if self.boxes.size == 0:
            boxes, covs = [], []
            for idx, box in enumerate(new_boxes):
                mean, cov = PersistentBoxes.kalman_filter.initiate(self.xyxy2xywh(box))
                boxes.append(mean)
                covs.append(cov)

            self.boxes = np.stack(boxes, axis=0) if len(boxes) > 0 else np.empty((0, 8))
            self.cov = np.stack(covs, axis=0) if len(boxes) > 0 else np.empty((0, 8, 8))
            self.cls = new_cls
            self.timer = np.zeros((self.boxes.shape[0],), dtype=np.int32)

            return self.xywh2xyxy(self.boxes[:, :4]), self.cls
        
        self.boxes, self.cov = PersistentBoxes.kalman_filter.multi_predict(self.boxes, self.cov)  # predict new timestep locations

        if new_boxes.shape[0] == 0:
            self.timer += 1
            return self.xywh2xyxy(self.boxes[:, :4]), self.cls
        
        # kalman
        # multi_predict to predict new locations
        # match predicted with current detections
        # update any matched detections to get new mean and covariance

        # make sure (as is now) if new box is different cls, old box is overriden
        # perfer persistent boxes to low confidence artifacts and filtering
        
        nboxes = new_boxes[None, :, :]
        curr_boxes = self.boxes[:, None, :4]

        iou_grid = self.iou(self.xywh2xyxy(curr_boxes), nboxes)  # (n current, n new)

        # updates for ones that overlap tracked boxes
        argmax = iou_grid.argmax(axis=1, keepdims=True)
        overlap = np.take_along_axis(iou_grid, argmax, axis=1) > 0.3
        argmax = argmax[overlap].reshape(-1)  # (n)

        boxes, covs, cls, timer = [], [], [], []
        new_boxes = self.xyxy2xywh(new_boxes)
        
        for i, j in enumerate(argmax):
            mean, cov = PersistentBoxes.kalman_filter.update(self.boxes[i], self.cov[i], new_boxes[j])
            boxes.append(mean)
            covs.append(cov)
            cls.append(new_cls[j:j+1].reshape(-1))
            timer.append(np.zeros((1,), dtype=np.int32))

        # initiate for new ones that weren't updated
        new_idxs = np.arange(new_boxes.shape[0])
        new_idxs = np.setdiff1d(new_idxs, argmax)  # ones that weren't updated
        for idx in new_idxs:
            mean, cov = PersistentBoxes.kalman_filter.initiate(new_boxes[idx])
            boxes.append(mean)
            covs.append(cov)
            cls.append(new_cls[idx:idx+1].reshape(-1))
            timer.append(np.zeros((1,), dtype=np.int32))

        # persist for tracked boxes that aren't updated and aren't past their time limit
        persist = ~(overlap.reshape(-1)) & (self.timer < self.persistence)
        for box, cov, cls_, timer_ in zip(self.boxes[persist], self.cov[persist], self.cls[persist], self.timer[persist]):
            boxes.append(box)
            covs.append(cov)
            cls.append(cls_.reshape(-1))
            timer.append(timer_.reshape(-1) + 1)
        
        self.boxes = np.stack(boxes, axis=0) if len(boxes) > 0 else np.empty((0, 8))
        self.cov = np.stack(covs, axis=0) if len(boxes) > 0 else np.empty((0, 8, 8))
        self.cls = np.concatenate(cls, axis=0) if len(boxes) > 0 else np.empty((0,))
        self.timer = np.concatenate(timer, axis=0) if len(boxes) > 0 else np.empty((0,))

        return self.xywh2xyxy(self.boxes[:, :4]), self.cls


class PersistentDisplay():
    def __init__(self, n_zones=7, img_res=(960, 1280), persistence=30):
        self.zones = np.zeros((n_zones,), dtype=bool)
        self.cls = np.full((n_zones,), -1, dtype=np.int32)
        self.bins = np.linspace(0, img_res[1], num=n_zones + 1)
        self.timer = np.zeros((n_zones,), dtype=np.int32)

        self.c = 0

        self.h = img_res[0]
        self.w = img_res[1]
        self.persistence = persistence  # num updates to hold box
        self.cls_to_color = {-1: (0, 0, 0), 0: (0, 0, 255), 1: (0, 234, 255), 2: (0, 255, 0)}  # yellow: (0, 234, 255), currently just replacing with red

    def xyxy2xywh(self, boxes):
        wh = boxes[..., 2:4] - boxes[..., :2]
        xy = (boxes[..., 2:4] + boxes[..., :2]) / 2
        mins = np.array([0, 0, 0, 0])
        maxs = np.array([self.w, self.h, self.w, self.h])
        return np.clip(np.concatenate([xy, wh], axis=-1), a_min=mins, a_max=maxs)
    
    def update_color(self, img, boxes, cls):
        corrected_cls = np.full_like(cls, -1)
        # box_res = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        # out_res = np.array([self.w, self.h, self.w, self.h])
        # boxes = ((boxes / box_res) * out_res).astype(np.int32)  # rescale for display
        boxes = boxes.astype(np.int32)
        for idx, box in enumerate(boxes):
            img_section = img[box[1]:box[3], box[0]:box[2], :]
            # if at least 60% of pixels are near black (under 30), replace cls color with the color of pixels above 70 if they are approximately green or red
            max_rgb = img_section.max(axis=-1)
            pct_black = max_rgb[max_rgb <= 25].size / max_rgb.size
            new_color = img_section[max_rgb > 70]
            t = new_color.mean(axis=0)
            # if not t[0] > t[1] + 70 and not t[1] > t[0] + 70 and self.c >= 80:
            #     cv2.imwrite('./test_tracking/section.png', img_section[..., ::-1])
            #     print(img_section[..., 0])
            #     print(img_section[..., 1])
            #     print(img_section[..., 2])
            #     print(new_color.mean(axis=0))
            #     assert False
            if new_color.size > 0:  # pct_black >= 0.60 and 
                new_color = new_color.mean(axis=0)
                if new_color[0] > new_color[1] + 60:  # red brighter than others by at least 70
                    corrected_cls[idx] = 0
                elif new_color[1] > new_color[0] + 60: # same for green
                    corrected_cls[idx] = 2
        
        return corrected_cls

    def update(self, new_boxes, new_cls, img):
        xyxy = new_boxes.numpy()
        new_boxes = self.xyxy2xywh(xyxy.copy())
        new_cls = new_cls.numpy()

        activate_zones = np.digitize(new_boxes[:, 0], self.bins) - 1
        deactivate_zones = self.timer >= self.persistence
        self.timer += 1

        self.zones[activate_zones] = True

        highest_box = []
        for zidx in np.unique(activate_zones):
            msk = activate_zones == zidx
            box_cpy = new_boxes.copy()
            box_cpy[~msk] = np.inf
            closest_box = box_cpy[:, 1].argmin()  # box that is highest vertically stays (heuristic for light being closer)
            highest_box.append(closest_box)
        highest_box = np.array(highest_box, dtype=np.int32)

        corrected_cls = np.full_like(self.cls, -1)
        corrected_cls[activate_zones[highest_box]] = self.update_color(img, xyxy[highest_box], new_cls[highest_box])
        self.cls[activate_zones[highest_box]] = new_cls[highest_box]
        self.timer[activate_zones] = 0

        deactivate_zones[activate_zones] = False
        self.zones[deactivate_zones] = False
        self.cls[deactivate_zones] = -1
        self.cls[corrected_cls != -1] = corrected_cls[corrected_cls != -1]

        # turn into image
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for i in range(self.zones.shape[0]):
            height = int(self.h / 2) if corrected_cls[i] != -1 else self.h
            cv2.rectangle(img, (int(self.bins[i]), 0), (int(self.bins[i + 1]), height), color=self.cls_to_color[self.cls[i]], thickness=cv2.FILLED)

        return img


def load_bosch():
    with open('bosch_test/img_paths.txt', 'r') as sequence_files:  # test_tracking/images.txt   lisa/val_images.txt
        sequences = [sequence.strip() for sequence in sequence_files.readlines()]

    sequences.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    return sequences

def load_driveu():
    with open('driveu/bremen.txt', 'r') as sequence_files:  # test_tracking/images.txt   lisa/val_images.txt
        sequences = [sequence.strip() for sequence in sequence_files.readlines()]

    sequences.sort()
    return sequences

def load_lisa():
    with open('lisa/val_images.txt', 'r') as sequence_files:  # test_tracking/images.txt   lisa/val_images.txt
        sequences = [sequence.strip() for sequence in sequence_files.readlines()]

    day, night = [], []
    for pth in sequences:
        if pth.startswith('day'): day.append(pth)
        else: night.append(pth)
    
    day.sort(key=lambda x: int(x.split('--')[-1].split('.')[0]))
    night.sort(key=lambda x: int(x.split('--')[-1].split('.')[0]))

    return day + night

def viz_seq(start=0, end=250, data='lisa'):
    fps = 10
    if data == 'lisa': sequences = load_lisa()
    elif data == 'bosch': sequences = load_bosch()
    else: 
        sequences = load_driveu()
        fps = 2

    h, w = cv2.imread(sequences[start]).shape[:2]
    print(h, w)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('test_tracking/out.mp4', fourcc, fps, (w, h))
    display = cv2.VideoWriter('test_tracking/display.mp4', fourcc, fps, (w, h))  # what will be displayed on the screen

    persistent_display = PersistentDisplay(n_zones=9, img_res=(h, w), persistence=int(0.8 * fps))  # (960, 1280)

    for i in range(start, end):
        out = model.track(sequences[i], persist=True)[0]  # list of len batch
        img = torch.from_numpy(out.orig_img).permute(2, 0, 1).flip(0)

        new_boxes = out.boxes.xyxy.cpu()
        cls = out.boxes.cls.cpu()
        update_display = persistent_display.update(new_boxes, cls, img.permute(1, 2, 0).numpy())

        img_with_bbox = torchvision.utils.draw_bounding_boxes(img, new_boxes, colors=[cls_to_color[int(cls_)] for cls_ in cls]) if new_boxes.shape[0] > 0 else img
        # res = cv2.imwrite(f'test_tracking/{i}.jpg', img_with_bbox.flip(0).permute(1, 2, 0).numpy())
        video.write(img_with_bbox.flip(0).permute(1, 2, 0).numpy())
        display.write(update_display)

    video.release()
    display.release()

# qualitative results on night tracking
# metrics when return

def xywh2xyxy(bbox):
    wh_ = bbox[..., 2:] / 2
    xyxy = torch.cat([bbox[..., :2] - wh_, bbox[..., :2] + wh_], dim=-1)
    return xyxy


def iou(boxes, gt, eps=1e-5):
    pmins = boxes[..., :2]
    pmaxs = boxes[..., 2:]
    gtmins = gt[..., :2]
    gtmaxs = gt[..., 2:]

    intersection = (torch.minimum(pmaxs, gtmaxs) - torch.maximum(pmins, gtmins)).clamp(0)
    intersection = intersection[..., 0] * intersection[..., 1]  # intersection area

    parea, gtarea = (pmaxs - pmins), (gtmaxs - gtmins)
    union = parea[..., 0] * parea[..., 1] + gtarea[..., 0] * gtarea[..., 1] - intersection + eps
    iou = intersection / union

    pxy = (pmaxs + pmins) / 2
    xy = (gtmaxs + gtmins) / 2
    xydiff = torch.abs(xy - pxy)

    return iou, xydiff


def confusion_matrix(data='bosch'):
    if data == 'bosch': img_paths = load_bosch()
    else: img_paths = load_driveu()

    seq_length = 200
    while True:
        ranges = torch.randint(0, len(img_paths) - seq_length, (2,)).sort()[0]
        if not torch.any((ranges[1:] - ranges[:-1]) < seq_length):  # make sure further apart than sequence length
            break

    # ranges = torch.tensor([0, 591, 1424], dtype=torch.int32)
    # ranges = torch.tensor([22011], dtype=torch.int32)

    tp, fp, fn, fp_right_cls, fn_wrong_cls = 0, 0, 0, 0, 0
    for range_start in ranges:
        for i in range(range_start, range_start + seq_length):
            annot_path = img_paths[i][:-4] + '.txt'
            # annot_path = f'train/labels/{i}.txt'
            with open(annot_path) as annot:
                labels = [line.strip().split(' ') for line in annot.readlines()]
                labels = [[float(coord) for coord in label[1:]] + [int(label[0])] for label in labels] # xywh normalized
                labels = torch.tensor(labels).view(-1, 5)
                labels[:, 2:4].clamp_(max=torch.min(2 * (labels[:, :2]), 2 * (1 - labels[:, :2])))

            img_path = img_paths[i]
            bboxes, labels = labels.split((4, 1), dim=1)
            labels = labels.int()
            persist = i != 0
            tracking_out = model.track(img_path, persist=persist)[0]  # list of len batch

            orig_shape = torch.tensor(tracking_out.orig_img.shape)
            pred_bboxes = tracking_out.boxes.xyxy.cpu()
            bboxes = xywh2xyxy(bboxes) * orig_shape[[1, 0, 1, 0]]  # orig whwh

            npred, ngt = pred_bboxes.shape[0], bboxes.shape[0]
            if npred == 0:
                tp += 0
                fp += 0
                fn += ngt
                fp_right_cls += 0
                fn_wrong_cls += 0
                continue
            if ngt == 0:
                tp += 0
                fp += npred
                fn += 0
                fp_right_cls += 0
                fn_wrong_cls += 0
                continue

            iou_grid, xydiff = iou(pred_bboxes[:, None, :].expand(-1, ngt, -1), bboxes[None, :, :].expand(npred, -1, -1))  # (preds, gt) iou
            cls_mask = tracking_out.boxes.cls.cpu().view(-1).to(dtype=torch.int32)[:, None] == labels.squeeze(1)[None, :]

            matching = (iou_grid > 0.1) # (iou_grid > 0.1)  # torch.linalg.norm(xydiff, dim=-1) < 50
            tps = (matching & cls_mask).sum(dim=0).bool().sum()
            tp += tps
            fp += npred - tps
            fn += ngt - tps
            dist = torch.linalg.norm(xydiff, dim=-1)
            mins = dist.min(dim=1, keepdim=True)[0]
            closest_gt_mask = dist == mins
            fp_right_cls += (~matching & cls_mask & closest_gt_mask).sum(dim=1).bool().sum()  # for each pred, if it wasn't matching but predicted the right cls for the gt closest to it
            fn_wrong_cls += (matching & (~cls_mask)).sum(dim=0).bool().sum()  # for each gt, if any pred has > 0.1 iou and predicted the wrong cls

    row = f'{tp}   |   {fn}'
    print('-'*len(row))
    print(row)
    print('-'*len(row))
    print(f'{fp.item()}  |')
    print('-'*len(row))
    print(fp_right_cls.item())
    print('-'*len(row))
    print(fn_wrong_cls.item())
    print('-'*len(row))


if __name__ == '__main__':
    # viz_seq(start=2236, end=2500, res=(960, 1280))  # night sequence for val
    start = 3080  # 3530
    viz_seq(start=start, end=start+200, data='lisa')

    # model.to('cuda')
    # out = model.track([np.random.rand(*(960, 1280, 3))], persist=True)[0]
    # print(len(out))

    # confusion_matrix(data='driveu')

    pass


'''
may have to modify frame rate in tracker/track.py
conf threshold in engine/model.py
'''
