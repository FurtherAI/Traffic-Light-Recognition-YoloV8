import torch
import torch.nn.functional as F
import numpy as np
from lisa import LISA
from torch.utils.data import DataLoader
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler_pytorch import WarmUpScheduler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import os

from yolov8 import DetectionModel, Detect
from ema import ModelEMA


class YOLO(pl.LightningModule):
    def __init__(self, cfg='yolov8.yaml', pretrained=False, freeze=False, init_resolution=(416, 416), yolo_reduction_factor=32):
        super().__init__()
        self.save_hyperparameters()
        self.pretrained = pretrained
        self.freeze = freeze
        self.init_resolution = init_resolution
        self.yolo_reduction_factor = yolo_reduction_factor
        self.yolo = DetectionModel(cfg=cfg, ch=3, nc=3)

        self.yolo.criterion = self.yolo.init_criterion()

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
    
    def xywh2xyxy(self, xywh):
        xy, wh = xywh.chunk(2, dim=-1)
        _wh = wh / 2
        xy, _xy = xy - _wh, xy + _wh
        return torch.cat([xy, _xy], dim=-1)  # now in xyxy

    def forward(self, x):
        return self.yolo(x)

    def training_step(self, batch, batch_idx, train=True):
        # torch.cuda.empty_cache()
        images, labels, _, _ = batch
        preds = self(images)

        loss = self.yolo.criterion(preds, batch)

        if train:
            self.log('loss', loss)

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


def pretrained(yolo, anchors, load_pt=True, load_ckpt=False, freeze=True):
    if load_pt:
        yolo = torch.load('yolov8s/yolov8s.pt')  # load pretrained weights
    elif load_ckpt:
        yolo = yolo.load_from_checkpoint('yolov8s/last.ckpt')

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
    batch_size = 32
    train_data = LISA(split='train')
    val_data = LISA(split='val')

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

    yolo = YOLO(pretrained=False, freeze=False, init_resolution=(416, 416), yolo_reduction_factor=32)
    yolo = pretrained(yolo, load_pt=True, load_ckpt=False, freeze=False)

    trainer =  pl.Trainer(
        gradient_clip_val=5,
        accelerator='gpu',
        auto_select_gpus=True,
        benchmark=True,  # should be faster for constant size batches
        max_epochs=10,
        limit_val_batches=0.2,
        # overfit_batches=1,
        # profiler="simple", 
        callbacks=[ckpt_callback],
        default_root_dir='logs/',
        log_every_n_steps=10,
    )

    trainer.fit(yolo, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)  # ckpt_path='yolov3-tiny/last.ckpt', 

    torch.save(yolo, 'yolov8s/v8_!ema.pt')

    # yolo detect train model=yolov8s.pt pretrained=True data=ultralytics/dataset.yaml epochs=12 batch=32 imgsz=416 device=0 optimizer=AdamW cos_lr=True label_smoothing=0.01 lr0=0.0001 warmup_epochs=0.1

    