import torch
import pytorch_lightning as pl
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import cv2
# from lisa_dataset import LISA, MEAN, STD
from lisa import LISA
from train import YOLO
import os
from metrics import ap_per_class


def viz(yolo, data):
    idx = 31
    # for idx in range(0, 32):
    #     batch = data.collate_fn([data[i] for i in range(idx, idx + 1)])
    #     yolo.compute_loss(yolo(batch[0]), *batch[1:])

    batch = data.collate_fn([data[i] for i in range(idx, idx + 1)])
    preds = yolo.nms(yolo(batch[0]))
    print(preds.shape)
    print(batch[0][0].shape)

    int_to_label = {0: 'stop', 1: 'warning', 2: 'go'}
    _, _, labels = data[idx]
    print('gt label:', int_to_label[labels.item()])
    print('pred label:', [int_to_label[box[4:7].argmax().item()] for box in preds])
    img = batch[0][0]
    img = ((img * torch.tensor(STD).view(-1, 1, 1) * 255) + (torch.tensor(MEAN).view(-1, 1, 1) * 255)).clamp(0, 255)
    img = img.to(dtype=torch.uint8)

    img_with_bbox = torchvision.utils.draw_bounding_boxes(img, preds[:, :4], colors=(255, 255, 0))
    res = cv2.imwrite('out.jpg', img_with_bbox.permute(1, 2, 0).numpy())
    print(res)

def unique_detections(iou_argmax, iou_max):
    counts = torch.zeros((iou_argmax.max() + 1,), dtype=torch.int32)  # counts location for every iou_argmax value
    unique_det = torch.zeros_like(iou_argmax, dtype=torch.bool)
    argsort = iou_max.argsort(descending=True)
    iou_argmax = iou_argmax[argsort]  # sort by max iou
    for idx, det in enumerate(iou_argmax):
        if counts[det] == 0: unique_det[argsort[idx]] = 1
        counts[det] += 1

    return unique_det

def metrics(yolo, data):
    # tp, conf, pred_cls, target_cls
    yolo = yolo.cuda()
    dataloader = DataLoader(
        data, 
        batch_size=64,
        collate_fn=data.collate_fn,
        num_workers=os.cpu_count(),
    )
    print(len(dataloader))
    tps, target_clss, pred_clss, confs = [], [], [], []
    for batch in dataloader:
        # bbox26 (batch, n, n, 4)
        # labels26 (batch, n, n)
        images, bbox13, labels13, bbox26, labels26 = batch
        images = images.to(device='cuda')
        batch_preds = yolo(images)
        del images
        batch_preds[0] = batch_preds[0].to(device='cpu')
        batch_preds[1] = batch_preds[1].to(device='cpu')

        for batch_idx in range(batch_preds[0].shape[0]):
            preds = yolo.nms([batch_preds[0][batch_idx:batch_idx + 1], batch_preds[1][batch_idx:batch_idx + 1]])  # (n, 7), cls already multiplied by objectness
            if preds.shape[0] == 0: continue  # false negatives, but these aren't used in metrics

            gt = bbox26[batch_idx][(labels26[batch_idx] != -1).unsqueeze(2).expand(-1, -1, 4)].reshape(-1, 4)
            gt_labels = labels26[batch_idx][labels26[batch_idx] != -1].view(-1)
            ngt, npred = gt.shape[0], preds.shape[0]

            pred_cls = preds[:, 4:7].argmax(dim=1)
            conf = preds[:, 4:7].take_along_dim(pred_cls.unsqueeze(1), dim=1).squeeze(1)

            if gt.shape[0] == 0:
                tp = torch.zeros((preds.shape[0],), dtype=torch.int32)
                target_cls = pred_cls.clone()  # mess with metrics the least if FP has no class error
            else:
                # generate iou matrix between each prediction and each gt box to find the best match
                rpreds = preds.repeat_interleave(ngt, dim=0)  # box1, box1, box1..., box2, box2, ...
                rgt = gt.repeat(npred, 1)  # box1, box2, ...
                iou = yolo.iou(rpreds[:, :4], rgt).view(npred, ngt)  # pred to gt iou matrix
                
                # determine true positives and which detections are lower iou boxes of the same gt (duplicates)
                iou_argmax = iou.argmax(dim=1)
                iou_max = iou.take_along_dim(iou_argmax.unsqueeze(1), dim=1).squeeze(1)
                unique_det = unique_detections(iou_argmax, iou_max)  # mask for which detections are the best per gt box
                tp = (iou_max > 0.5) & unique_det # mAP at 0.5 iou
                target_cls = gt_labels[iou_argmax]

            tps.append(tp)
            target_clss.append(target_cls)
            pred_clss.append(pred_cls)
            confs.append(conf)
            # should all be shape (n) number predicted boxes
        
        del bbox13
        del labels13
        del bbox26
        del labels26
        del batch_preds
        torch.cuda.empty_cache()
    
    tp = torch.cat(tps, dim=0).unsqueeze(1).to(dtype=torch.int32)
    target_cls = torch.cat(target_clss, dim=0)
    pred_cls = torch.cat(pred_clss, dim=0)
    conf = torch.cat(confs, dim=0)
    tp, fp, p, r, f1, ap, unique_classes = ap_per_class(tp, conf, pred_cls, target_cls, plot=False)
    ap = [ap[0] for p in ap]
    print('tp:', tp, '| fp:', fp, '| precision:', p, '| recall:', r, '| f1', f1, '| ap:', ap)
    print('mAP:', sum(ap) / len(ap))

def detections(yolo, data):
    # python3 pascalvoc.py -gt ./groundtruths/groundtruths -det ./detections/detections -gtformat xyrb -detformat xyrb -sp ./metrics_plots
    # class conf x1 y1 x2 y2
    int_to_label = {0: 'stop', 1: 'warning', 2: 'go'}
    yolo = yolo.cuda()
    dataloader = DataLoader(
        data, 
        batch_size=64,
        collate_fn=data.collate_fn,
        num_workers=os.cpu_count(),
    )
    print(len(dataloader))
    for data_idx, batch in enumerate(dataloader):
        # bbox26 (batch, n, n, 4)
        # labels26 (batch, n, n)
        images, bbox13, labels13, bbox26, labels26 = batch
        images = images.to(device='cuda')
        batch_preds = yolo(images)
        del images
        batch_preds[0] = batch_preds[0].to(device='cpu')
        batch_preds[1] = batch_preds[1].to(device='cpu')

        for batch_idx in range(batch_preds[0].shape[0]):
            preds = yolo.nms([batch_preds[0][batch_idx:batch_idx + 1], batch_preds[1][batch_idx:batch_idx + 1]])  # (n, 7), cls already multiplied by objectness
            with open(f'Object-Detection-Metrics/detections/detections/{data_idx * 64 + batch_idx}.txt', 'w') as out:
                for pred in preds:
                    out.write(f'{int_to_label[pred[4:7].argmax().item()]} {pred[4:7].max().item():.5f} {int(pred[0].item())} {int(pred[1].item())} {int(pred[2].item())} {int(pred[3].item())}\n')

        del bbox13
        del labels13
        del bbox26
        del labels26
        del batch_preds
        torch.cuda.empty_cache()


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


if __name__ == '__main__':
    anchors = [[25, 43],
               [43, 25]]
    anchors = torch.tensor(anchors, dtype=torch.float32, device='cpu')
    
    val_data = LISA(split='val')
    yolo = YOLO(anchors, init_resolution=(416, 416), yolo_reduction_factor=32, obj_label_smoothing=0.005, cls_label_smoothing=0.01)

    # yolo = yolo.load_from_checkpoint('yolov3-tiny/last.ckpt')
    # # yolo = yolo.cuda()
    # yolo.eval()
    # yolo.train()
    
    # detections(yolo, val_data)

    import yolov3.val as validate
    import yaml
    with open('/home/further/TLR/ultralytics/dataset.yaml', 'r') as file:
        data = yaml.safe_load(file)
    model = torch.load('yolov3-tiny/v5_training_!ema.pt').cuda().eval()
    model.names = ['stop', 'warning', 'go']
    # model.model = model.yolo

    # m, s = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1), torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    # x = LISA(split='val')
    # b1, b2 = x[1], x[0]
    # batch = x.collate_fn([b1, b2])
    # model = model.cpu()
    # model.anchors = model.anchors.cpu()

    grid = (416 // 32) * 2
    bias = (torch.arange(0, grid, device=anchors.device) - 0.5)
    model.grid = torch.stack([bias.view(1, 1, 1, grid).expand((-1, -1, grid, -1)), bias.view(1, 1, grid, 1).expand((-1, -1, -1, grid))], dim=-1)
    model.grid = model.grid.to(device=model.anchors.device)

    
    # preds = model(batch[0])
    # preds = model.nms(preds)
    # print(preds.shape)
    # print(preds[:, :-3])
    
    # img = (b1[0] * 255).to(dtype=torch.uint8)
    # img_with_bbox = torchvision.utils.draw_bounding_boxes(img, preds[:, :4], colors=(255, 255, 0))
    # res = cv2.imwrite('out.jpg', np.flip(img_with_bbox.permute(1, 2, 0).numpy(), axis=-1))
    # print(res)


    # assert False

    results, maps, _ = validate.run(data,
                                    batch_size=32,
                                    imgsz=416,
                                    model=model,
                                    half=False,
                                    single_cls=False,
                                    plots=False)
    


