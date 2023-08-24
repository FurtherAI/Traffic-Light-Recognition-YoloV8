import torch
import torch.nn.functional as F
import numpy as np
# from lisa_dataset import LISA
from lisa import LISA
from torch.utils.data import DataLoader
import yolov3.val as validate
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler_pytorch import WarmUpScheduler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os

from yolo import DetectionModel, Detect
from ema import ModelEMA


class YOLO(pl.LightningModule):
    def __init__(self, anchors, cfg='yolov3-tiny.yaml', pretrained=False, freeze=False, init_resolution=(416, 416), yolo_reduction_factor=32, obj_label_smoothing=0.005, cls_label_smoothing=0.05):
        super().__init__()
        self.save_hyperparameters()
        self.anchors = anchors  # (2, 2)
        self.pretrained = pretrained
        self.freeze = freeze
        self.init_resolution = init_resolution
        self.yolo_reduction_factor = yolo_reduction_factor
        self.obj_label_smoothing = obj_label_smoothing
        self.yolo = DetectionModel(cfg=cfg, ch=3, nc=3, anchors=anchors)

        self.mse = torch.nn.MSELoss()
        self.obj_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.cls_loss = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=cls_label_smoothing)

        grid = (init_resolution[0] // yolo_reduction_factor) * 2
        bias = (torch.arange(0, grid, device=anchors.device) - 0.5)
        self.grid = torch.stack([bias.view(1, 1, 1, grid).expand((-1, -1, grid, -1)), bias.view(1, 1, grid, 1).expand((-1, -1, -1, grid))], dim=-1)

    def pred_to_coord(self, boxes):
        grid = boxes.shape[-2]
        ai = 0 if grid == 26 else 1
        cxy = torch.stack([torch.arange(0, grid, device=boxes.device).view(1, 1, 1, grid).expand((-1, -1, grid, -1)), torch.arange(0, grid, device=boxes.device).view(1, 1, grid, 1).expand((-1, -1, -1, grid))], dim=-1)
        pcxy = torch.sigmoid(boxes[..., :2])
        boxes[..., :2] = (pcxy + cxy) * (self.init_resolution[0] // grid)  # now in pixels, center x and y (32, 3, n, n, 2)
        anchors = self.anchors[ai].view(1, self.anchors.shape[1], 1, 1, 2)
        boxes[..., 2:] = anchors * torch.exp(boxes[..., 2:])
        boxes[..., :2], boxes[..., 2:] = boxes[..., :2] - (boxes[..., 2:] / 2), boxes[..., :2] + (boxes[..., 2:] / 2)  # center x and y + or - width or height / 2
        # now in xmin, ymin, xmax, ymax form
        return boxes

    def nms(self, preds):
        # preds [(1, anchors, 2n, 2n, 8), (1, anchors, n, n, 8)]
        preds[0] = preds[0][0:1]
        preds[1] = preds[1][0:1]
        with torch.inference_mode():
            preds[0][..., 1:5], preds[1][..., 1:5] = self.pred_to_coord3(preds[0][..., 1:5]), self.pred_to_coord3(preds[1][..., 1:5])
            bboxes = torch.cat([preds[0].view(-1, 8), preds[1].view(-1, 8)], dim=0)
            torch.sigmoid_(bboxes[:, 0])
            bboxes[:, 5:] = F.softmax(bboxes[:, 5:], dim=1)

            bboxes[:, 5:] *= bboxes[:, 0:1]  # multiply iou * pr(object) * pr(class | object) = iou * pr(class and object)
            bboxes = bboxes[bboxes[:, 0] > 0.01][:, 1:]  # filter out low confidence boxes
            out = []
            for class_idx in range(4, 7):  # class probability score indexes
                boxes = bboxes.clone()
                boxes = boxes[boxes[:, 4:7].argmax(dim=1) == (class_idx - 4)]
                while not torch.all(boxes[:, class_idx] == 0) and boxes.shape[0] > 0:
                    best_box = boxes[boxes[:, class_idx].argmax()]
                    out.append(best_box)  # add max conf
                    iou_mask = self.iou(boxes[:, :4], best_box[None, :4]) <= 0.6
                    boxes = boxes[iou_mask]

            # stack to tensor before returning
            out = torch.stack(out[:10], dim=0) if len(out) > 0 else torch.empty((0, 4))
        return out

    def iou(self, boxes, gt, eps=1e-5):
        pmins = boxes[..., :2]
        pmaxs = boxes[..., 2:]
        gtmins = gt[..., :2]
        gtmaxs = gt[..., 2:]

        intersection = (torch.minimum(pmaxs, gtmaxs) - torch.maximum(pmins, gtmins)).clamp(0)
        intersection = intersection[..., 0] * intersection[..., 1]  # intersection area

        parea, gtarea = (pmaxs - pmins), (gtmaxs - gtmins)
        union = parea[..., 0] * parea[..., 1] + gtarea[..., 0] * gtarea[..., 1] - intersection + eps
        iou = intersection / union
        return iou

    def compute_loss(self, preds, bbox13, labels13, bbox26, labels26):
        # preds [(32, anchors, 2n, 2n, 8), (32, anchors, n, n, 8)]
        # bbox (32, n, n, 4)
        # labels (32, n, n)
        bboxes = [bbox26, bbox13]
        labels = [labels26, labels13]

        obj_loss, coord_loss, cls_loss = torch.zeros(1, device=bbox26.device), torch.zeros(1, device=bbox26.device), torch.zeros(1, device=bbox26.device)
        for res_idx in range(2):  # resolution index (for 13 vs 26)
            pred = preds[res_idx]
            obj_pred = pred[..., 0]
            bbox_pred = pred[..., 1:5]
            cls_pred = pred[..., 5:]
            bbox = bboxes[res_idx]
            label = labels[res_idx]
            grid_mask = label != -1

            bbox_pred = self.pred_to_coord(bbox_pred)
            iou = self.iou(bbox_pred, bbox.unsqueeze(1))  # (32, anchors, n, n)
            best_anchor = iou.argmax(dim=1, keepdim=True)

            # objectness
            # iou[iou == 0] = self.obj_label_smoothing

            obj = self.obj_loss(obj_pred, iou.detach())  # (32, 2, n, n)
            grid_anchor_mask = grid_mask.unsqueeze(1).expand(-1, self.anchors.shape[1], -1, -1)
            obj_loss += torch.nan_to_num(obj[grid_anchor_mask].mean()) + obj[~grid_anchor_mask].mean()  # obj.mean()

            # coordinate
            # responsible_ious = iou.take_along_dim(best_anchor, dim=1).squeeze(1)[grid_mask]  # box with maximum iou, only in grid locations with a ground truth bounding box
            responsible_ious = iou[grid_anchor_mask]
            # if grid mask is empty, coord loss will be empty and mean will be nan, replace with 0
            coord_loss += torch.nan_to_num((1 - responsible_ious).mean(), nan=0.0) # + 0.05 * F.mse_loss(bbox_pred, bbox.unsqueeze(1).expand(-1, 3, -1, -1, -1), reduction='none')[grid_anchor_mask].mean()

            # classification loss (using cross entropy, which is exclusive, because the light is one and only one color)
            label[label == -1] = 0
            cls = self.cls_loss(cls_pred.permute(0, 4, 1, 2, 3), label.unsqueeze(1).expand(-1, pred.shape[1], -1, -1))
            # cls = self.cls_loss(cls_pred.take_along_dim(best_anchor.unsqueeze(-1), dim=1).squeeze(1).permute(0, 3, 1, 2), label)  # best anchor per grid cell
            cls_loss += torch.nan_to_num(cls[grid_mask.unsqueeze(1).expand(-1, pred.shape[1], -1, -1)].mean(), nan=0.0)  # all anchors per each grid cell

        return obj_loss, coord_loss, cls_loss
    
    def pred_to_coord2(self, boxes, anchors):
        # boxes and anchors are on grid scale
        # not recentering because targets are normalized to the index center they have
        pxy, pwh = boxes.chunk(2, dim=1)
        pxy = pxy.sigmoid() * 2 - 0.5
        pwh = (pwh.sigmoid() * 2) ** 2 * anchors
        pbox = torch.cat((pxy, pwh), dim=1)
        return pbox
    
    def pred_to_coord3(self, boxes):
        red = self.init_resolution[0] // boxes.shape[-2]
        stride = torch.tensor((red,), dtype=torch.int32, device=boxes.device)  # pixels per grid cell
        ai = 0 if red == 16 else 1
        anchor_grid = self.anchors[ai].view(1, self.anchors.shape[1], 1, 1, self.anchors.shape[2]).expand(-1, -1, boxes.shape[-2], boxes.shape[-2], -1) # anchors shaped like grid
        grid = self.grid[:, :, :boxes.shape[-2], :boxes.shape[-2], :]

        xywh = boxes.clone()

        pxy, pwh = xywh.chunk(2, dim=-1)
        pxy = (pxy.sigmoid() * 2 + grid) * stride
        pwh = (pwh.sigmoid() * 2) ** 2 * anchor_grid
        pbox = torch.cat((pxy, pwh), dim=-1)

        pbox = self.xywh2xyxy(pbox)
        return pbox
    
    def xywh2xyxy(self, xywh):
        xy, wh = xywh.chunk(2, dim=-1)
        _wh = wh / 2
        xy, _xy = xy - _wh, xy + _wh
        return torch.cat([xy, _xy], dim=-1)  # now in xyxy
    
    def compute_loss2(self, preds, targets):
        obj_loss, coord_loss, cls_loss = torch.zeros(1, device=preds[0].device), torch.zeros(1, device=preds[0].device), torch.zeros(1, device=preds[0].device)
        tcls, tbox, indexes, anch = self.build_targets(preds, targets)

        obj_balance = (4.0, 1.0)
        for idx, pred in enumerate(preds):
            b, a, i, j = indexes[idx]
            tobj = torch.zeros_like(pred[..., 0])

            _, pxywh, pcls = pred[b, a, i, j].split((1, 4, 3), dim=1)

            # bbox loss
            pbox = self.pred_to_coord2(pxywh, anch[idx])
            # tbox center already normalized to correct grid location in build_targets

            iou = self.iou(self.xywh2xyxy(pbox), self.xywh2xyxy(tbox[idx]))
            coord_loss += (1.0 - iou).mean() # + 0.2 * F.mse_loss(pbox, tbox[idx], reduction='none').mean()

            # objectness
            iou = iou.detach()
            tobj[b, a, i, j] = iou
            obj_loss += obj_balance[idx] * self.obj_loss(pred[..., 0], tobj).mean()
            # print(obj_loss)

            # classification
            cls_loss += self.cls_loss(pcls, tcls[idx]).mean()
            # print(cls_loss)
        
        return torch.nan_to_num(obj_loss, 0.0), torch.nan_to_num(coord_loss, 0.0), torch.nan_to_num(cls_loss, 0.0)  # if labels is len 0

    def build_targets(self, preds, targets):
        na, nt = self.anchors.shape[1], targets.shape[0]
        tcls, tbox, indexes, anch = [], [], [], []
        scale = torch.ones(7, device=targets.device)
        ai = torch.arange(na, device=targets.device).view(-1, 1).expand(-1, nt)
        targets = torch.cat((targets.unsqueeze(0).expand(na, -1, -1), ai.view(na, nt, 1)), dim=2)  # na, nt, 7 (b, c, x, y, w, h, a)

        off = torch.tensor(
        [
            [0, 0],
            [-1, 0],
            [0, -1],
            [1, 0],
            [0, 1]
        ], device=targets.device)
        for i in range(2):
            anchors, grid = self.anchors[i].clone(), preds[i].shape[-2]
            anchors /= (self.init_resolution[0] // grid)  # normalize to grid size
            scale[2:6] = grid

            t = targets * scale  # now normalized to grid
            r = t[..., 4:6] / anchors.unsqueeze(1)
            msk = torch.max(r, 1 / r).max(dim=2)[0] < 4  # make sure target is less than 4x anchor width/height
            t = t[msk]  # (n, 7)

            # repeat targets for grid cells around the original target (within 0.5 range), any cell that could predict the right center gets a target there
            gxy = t[:, 2:4]
            gxyi = grid - gxy
            j, i = ((gxy % 1 < 0.5) & (gxy > 1)).T
            _j, _i = ((gxyi % 1 < 0.5) & (gxyi > 1)).T
            msk = torch.stack((torch.ones_like(j), j, i, _j, _i), dim=0)
            t = t.unsqueeze(0).expand(5, -1, -1)[msk]
            offsets = off.unsqueeze(1).expand(-1, gxy.shape[0], -1)[msk]
        
            bc, gxy, gwh, a = t.chunk(4, dim=1)
            a, (b, c) = a.long().squeeze(), bc.long().T
            gji = (gxy + offsets).long()
            gj, gi = gji.T

            indexes.append((b, a, gi.clamp_(0, grid - 1), gj.clamp_(0, grid - 1)))  # batch, anchor, row, col of grid
            tbox.append(torch.cat((gxy - gji, gwh), dim=1))
            anch.append(anchors[a])
            tcls.append(c)
        
        return tcls, tbox, indexes, anch
    
    def valforward(self, x):
        z = []
        preds = self.yolo(x)

        anchor_grid = [self.anchors[i].view(1, self.anchors.shape[1], 1, 1, self.anchors.shape[2]).expand(-1, -1, preds[i].shape[-2], preds[i].shape[-2], -1) for i in range(2)]  # anchors shaped like grid
        grid = [self.grid[:, :, :preds[i].shape[-2], :preds[i].shape[-2], :] + 0.5 for i in range(2)]

        for i in range(2):
            conf, xywh, c = preds[i].clone().split((1, 4, 3), dim=-1)
            preds[i] = torch.cat((xywh, conf, c), dim=-1)
            # also it expects outputs in xywh, full coords

            pcxy = torch.sigmoid(xywh[..., :2])
            xywh[..., :2] = (pcxy + grid[i]) * (self.init_resolution[0] // xywh.shape[-2])  # now in pixels, center x and y (32, 3, n, n, 2)
            xywh[..., 2:] = anchor_grid[i] * torch.exp(xywh[..., 2:])

            y = torch.cat((xywh, conf.sigmoid(), c), dim=-1)
            z.append(y.view(y.shape[0], -1, y.shape[-1]))

        return torch.cat(z, dim=1), preds
    
    def vforward(self, x):
        z = []
        preds = self.yolo(x)

        red = self.init_resolution[0] // preds[0].shape[-2]
        stride = torch.tensor((red, red << 1), dtype=torch.int32, device=preds[0].device)  # pixels per grid cell
        anchor_grid = [self.anchors[i].view(1, self.anchors.shape[1], 1, 1, self.anchors.shape[2]).expand(-1, -1, preds[i].shape[-2], preds[i].shape[-2], -1) for i in range(2)]  # anchors shaped like grid
        grid = [self.grid[:, :, :preds[i].shape[-2], :preds[i].shape[-2], :] for i in range(2)]

        for i in range(2):
            conf, xywh, c = preds[i].clone().split((1, 4, 3), dim=-1)
            preds[i] = torch.cat((xywh, conf, c), dim=-1)

            pxy, pwh = xywh.sigmoid().chunk(2, dim=-1)
            pxy = (pxy * 2 + grid[i]) * stride[i]
            pwh = (pwh * 2) ** 2 * anchor_grid[i]
            pbox = torch.cat((pxy, pwh), dim=-1)

            y = torch.cat((pbox, conf.sigmoid(), c), dim=-1)
            z.append(y.view(y.shape[0], -1, y.shape[-1]))

        return torch.cat(z, dim=1), preds

    def forward(self, x):
        return self.yolo(x)

    # def training_step(self, batch, batch_idx, train=True):
    #     # torch.cuda.empty_cache()
    #     images, bbox13, labels13, bbox26, labels26 = batch
    #     preds = self(images)
    #     obj_loss, coord_loss, cls_loss = self.compute_loss(preds, bbox13, labels13, bbox26, labels26)

    #     alpha, beta, gamma = 1.0, 0.5, 0.33
    #     loss = alpha * obj_loss + beta * coord_loss + gamma * cls_loss #

    #     if train:
    #         self.log('losses', {'obj': alpha * obj_loss, 'coord/iou': beta * coord_loss, 'cls': gamma * cls_loss})
    #         self.log('loss', loss)

    #     # Exponential moving average to improve validation
    #     if not hasattr(self, 'ema'):
    #         self.ema = ModelEMA(self)
    #     self.ema.update(self)

    #     return loss

    def training_step(self, batch, batch_idx, train=True):
        # torch.cuda.empty_cache()
        images, labels, _, _ = batch
        preds = self(images)
        obj_loss, coord_loss, cls_loss = self.compute_loss2(preds, labels)

        alpha, beta, gamma = 1.0, 0.05, 0.5
        loss = alpha * obj_loss + beta * coord_loss + gamma * cls_loss #

        if train:
            self.log('losses', {'obj': alpha * obj_loss, 'coord/iou': beta * coord_loss, 'cls': gamma * cls_loss})
            self.log('loss', loss)

        loss *= 32  # multiply by batch size (from ultralytics (for large batch training, but here to reproduce))
        return loss
    
    def training_step_end(self, outputs):
        lr = self.lr_schedulers().get_last_lr()[0]
        self.log('learning rate', lr)

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx, train=False)
        self.log('validation loss', loss)

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step()

    def configure_optimizers(self):
        # if freeze, freeze layers except detection head
        # if pretrained, batch norm in eval mode to not update running stats
        groups = [], []  # bias/batch norm parameters - no weight decay, weights - weight decay
        norms = tuple(v for k, v in torch.nn.__dict__.items() if 'Norm' in k)  # any torch layer that is a normalization layer
        head = False
        # DFS, so simple approach is to know we're in Detect once encountered, then out once we encounter a module after detect
        # don't have to worry about once we're out of detect though because it's the last module
        for mod in self.yolo.modules():
            if isinstance(mod, Detect): head = True
            if self.pretrained and isinstance(mod, norms):
                mod.eval()
            for param_name, param in mod.named_parameters(recurse=False): # module loop already handles recursion
                if self.freeze and not head:  # freeze everything except head
                    param.requires_grad_(False)
                if param_name == 'bias' or (param_name == 'weight' and isinstance(param, norms)):
                    groups[0].append(param)
                else:
                    groups[1].append(param)
        
        optimizer = torch.optim.AdamW([{'params': groups[0], 'weight_decay': 0.0}, 
                                       {'params': groups[1]}
                                       ], lr=1e-4, weight_decay=1e-5)
        
        total_steps = self.trainer.estimated_stepping_batches
        cosine_steps = int(0.9 * total_steps)
        warmup_steps = total_steps - cosine_steps

        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=5e-6, last_epoch=-1)
        lr_scheduler = WarmUpScheduler(optimizer, cosine_scheduler, len_loader=total_steps, warmup_steps=warmup_steps, warmup_start_lr=5e-6, warmup_mode='linear')
        return {"optimizer" : optimizer, 
                "lr_scheduler" : {
                    "scheduler" : lr_scheduler,
                    "interval" : "step",
                    "frequency" : 1
                }
        }


class ValidationCB(pl.Callback):
    def on_train_epoch_end(self, trainer, yolo):
        model = yolo.ema.ema if hasattr(yolo, 'ema') else yolo
        model.grid = model.grid.to(device='cuda')
        results, maps, _ = validate.run(yolo.data,
                                    batch_size=32,
                                    imgsz=416,
                                    model=model,
                                    half=False,
                                    single_cls=False,
                                    plots=False)


def pretrained(yolo, anchors, load_pt=True, load_ckpt=False, freeze=True):
    if load_pt:
        yolo = torch.load('yolov3-tiny/yolov3-tiny.pt')  # load pretrained weights
    elif load_ckpt:
        yolo = yolo.load_from_checkpoint('yolov3-tiny/last.ckpt')

    yolo.anchors = anchors.to(device='cuda')
    yolo.grid = yolo.grid.to(device='cuda')
    yolo.pretrained = True
    yolo.freeze = freeze

    with open('/home/further/TLR/ultralytics/dataset.yaml', 'r') as file:
        yolo.data = yaml.safe_load(file)

    yolo.names = ['stop', 'warning', 'go']

    yolo.count = 0

    return yolo

if __name__ == '__main__':
    # anchors = [[25, 43],
    #            [43, 25]]
    anchors = [[[3, 6],
                [5, 9],
                [8, 13]],
                [[10, 18],
                 [13, 23],
                 [17, 26]]]
    anchors = torch.tensor(anchors, dtype=torch.float32, device='cuda')

    batch_size = 32
    train_data = LISA(split='train')
    val_data = LISA(split='val')

    # train_data = LISA(split='val')
    train_dataloader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=train_data.collate_fn,
        num_workers=os.cpu_count(),
    )
    validation_dataloader = DataLoader(
        val_data, 
        batch_size=batch_size,
        collate_fn=val_data.collate_fn,
        num_workers=os.cpu_count(),
    )

    steps_per_epoch = len(train_dataloader)
    ckpt_callback = ModelCheckpoint(
        dirpath='yolov3-tiny/',
        save_last=True,
        save_weights_only=True,  # don't save lr, optimizers etc
        # monitor='validation loss',
        every_n_train_steps=int(steps_per_epoch / 20),
        save_on_train_epoch_end=True
    )
    val_callback = ValidationCB()

    yolo = YOLO(anchors, pretrained=False, freeze=False, init_resolution=(416, 416), yolo_reduction_factor=32, obj_label_smoothing=0.005, cls_label_smoothing=0.01)
    yolo = pretrained(yolo, anchors, load_pt=True, load_ckpt=False, freeze=False)

    trainer =  pl.Trainer(
        gradient_clip_val=5,
        accelerator='gpu',
        auto_select_gpus=True,
        benchmark=True,  # should be faster for constant size batches
        max_epochs=10,
        limit_val_batches=0.2,
        # overfit_batches=1,
        # profiler="simple", 
        callbacks=[ckpt_callback, val_callback],
        default_root_dir='logs/',
        log_every_n_steps=10,
    )

    trainer.fit(yolo, train_dataloaders=train_dataloader, )  # ckpt_path='yolov3-tiny/last.ckpt', val_dataloaders=validation_dataloader

    torch.save(yolo, 'yolov3-tiny/v5_training_ema.pt')
    '''
    They don't normalize with imagenet, just divide by 255
    don't forget to switch nms pred to coord and the dataset, in val.py too

    in val.py: training = False, ncm = 3, comment out warmup, stride = 32, ln 215 comment out compute_loss, utils dataloaders rect and mosaic = False, changed sorting of self.im_files in dataloaders,
    in dataloaders.py create dataloader changed to lisa and collate fn, in val, removed / 255, (lines 253, 258) scale_boxes, commented out path, shape, val plot and save dir,
    in val vforward instead of model(im), dataloaders ln 1037 duplicate labels, 
    train on 416 (lower) resolution, then scale up for fine tuning to save cost in training
    python3 train.py --weights ../yolov3-tiny.pt --cfg ../yolov3-tiny.yaml --data ../ultralytics/dataset.yaml --epochs 5 --batch-size 32 --imgsz 416 --device 0 --optimizer AdamW --cos-lr --label-smoothing 0.01
    '''

    