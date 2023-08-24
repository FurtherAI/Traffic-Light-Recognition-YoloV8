import os
import torch
import torchvision
import numpy as np
import cv2
import yaml
import json
from train import YOLO


def rm_rel_path(frm, to):
    with open(frm, 'r') as files:
        paths = files.readlines()
        paths = [path[2:] for path in paths]
        with open(to, 'w') as out:
            paths.sort()
            for path in paths:
                if path[:6] != 'sample': out.write(path)

def display_img_with_box(pth):
    img = cv2.imread(pth)
    img = torch.as_tensor(img, dtype=torch.uint8)
    img = img.permute(2, 0, 1)
    
    box_annot = 'dayTest/daySequence1--00000.jpg;stop;706;478;718;500;dayTest/daySequence1/Day1EveningShutter0.000200-Gain-0.mp4;0;dayTest/daySequence1/Day1EveningShutter0.000200-Gain-0.mp4;0'
    bulb_annot = 'dayTest/daySequence1--00000.jpg;stop;710;481;714;486;dayTest/daySequence1/Day1EveningShutter0.000200-Gain-0.mp4;0;dayTest/daySequence1/Day1EveningShutter0.000200-Gain-0.mp4;0'
    annot = box_annot.split(';')
    bbox = [int(coord) for coord in annot[2:6]]
    boxes = torch.tensor(bbox).view(-1, 4)
    
    img_with_bbox = torchvision.utils.draw_bounding_boxes(img, boxes, colors=(255, 255, 0))
    res = cv2.imwrite('out.jpg', img_with_bbox.permute(1, 2, 0).numpy())
    print(res)

def simplify_label(label):
    if label[:4] == 'stop':
        return 'stop'
    if label[:2] == 'go':
        return 'go'
    if label[:7] == 'warning':
        return 'warning'
    else:
        print(label)
        assert False

def create_annot_files():
    unique_labels = set()
    pth = {**{'dayClip' + str(i) : f'dayTrain/dayTrain/dayClip{i}/frames/' for i in range(1, 14)}, 
           **{'nightClip' + str(i) : f'nightTrain/nightTrain/nightClip{i}/frames/' for i in range(1, 6)},
           **{'daySequence' + str(i) : f'daySequence{i}/daySequence{i}/frames/' for i in range(1, 3)},
           **{'nightSequence' + str(i) : f'nightSequence{i}/nightSequence{i}/frames/' for i in range(1, 3)}}
    with open('box_files.txt', 'r') as box_files:
        for box_file in [line.strip() for line in box_files.readlines()]:
            with open(box_file, 'r') as annot_in:
                annots = [aline.strip() for aline in annot_in.readlines()[1:]]
                annot = ''
                annot_line = annots[0].split(';')[:6]
                prev_path = pth[annot_line[0].split('--')[0].split('/')[1]] + annot_line[0].split('/')[1][:-4] + '.txt'
                for annot_line in annots:
                    annot_line = annot_line.split(';')[:6]
                    unique_labels.add(annot_line[1])
                    annot_line[1] = simplify_label(annot_line[1])
                    curr_path = pth[annot_line[0].split('--')[0].split('/')[1]] + annot_line[0].split('/')[1][:-4] + '.txt'
                    
                    if curr_path != prev_path:
                        with open(prev_path, 'w') as annot_out: annot_out.write(annot)
                        annot = ''
                    annot += ','.join(annot_line) + '\n'
                    prev_path = curr_path
                with open(prev_path, 'w') as annot_out: annot_out.write(annot)

    print(unique_labels)

def make_train_test():
    with open('lisa/images.txt', 'r') as images:
        data_paths = [pth.strip() for pth in images.readlines()]
    train_mask = np.random.default_rng().binomial(n=1, p=0.8, size=(len(data_paths))).astype(bool)

    data_paths = np.array(data_paths, dtype=np.bytes_)
    train_examples = data_paths[train_mask]
    val_examples = data_paths[~train_mask]

    with open('train_images.txt', 'w') as train_out:
        for example in train_examples:
            train_out.write(example.decode('UTF-8') + '\n')
    with open('val_images.txt', 'w') as val_out:
        for example in val_examples:
            val_out.write(example.decode('UTF-8') + '\n')

def make_train_val_seq():
    with open('lisa/images.txt', 'r') as images:
        data_paths = [pth.strip() for pth in images.readlines()]

    with open('lisa/train_images.txt', 'w') as train_out, open('lisa/val_images.txt', 'w') as val_out:
        for data_path in data_paths:
            if data_path.startswith('daySequence1') or data_path.startswith('nightSequence1'):
                val_out.write(data_path + '\n')
            else:
                train_out.write(data_path + '\n')

def lisa_xyxy2nxywh(xmin, ymin, xmax, ymax):
    x, y, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, (xmax - xmin), (ymax - ymin)
    # resolution = (960, 1280)
    return x / 1280, y / 960, w / 1280, h / 960


def lisa_to_ultralytics(off, voff):
    # x, y, w, h = bosch_xyxy2nxywh(box['x_min'], box['y_min'], box['x_max'], box['y_max'])
    # annot_out.write(f'{label_to_int[cls]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')
    label_to_int = {'stop': 0, 'warning': 1, 'go': 2}
    with open('lisa/train_images.txt', 'r') as train_files, open('lisa/val_images.txt', 'r') as val_files:
        train_paths = [file.strip() for file in train_files]
        # lower the frequency of day train images. All images are very similar, but I want night so those can be kept at high frequency
        day_train = [pth for pth in train_paths if pth.startswith('day')][::3]
        night_train = [pth for pth in train_paths if pth.startswith('night')]

        val_paths = [file.strip() for file in val_files]
        day_val = [pth for pth in val_paths if pth.startswith('day')][::3]
        night_val = [pth for pth in val_paths if pth.startswith('night')]

        day_val_to_train = int(0.8 * len(day_val))
        night_val_to_train = int(0.8 * len(night_val))
        val_paths = day_val[day_val_to_train:] + night_val[night_val_to_train:]
        train_paths = day_train + night_train + day_val[:day_val_to_train] + night_val[:night_val_to_train]
    
    i = off
    while True:
        if os.path.exists(f'train/images/{i}.jpg'):
            os.remove(f'train/images/{i}.jpg')
            i += 1
        else: break

    i = voff
    while True:
        if os.path.exists(f'val/images/{i}.jpg'):
            os.remove(f'val/images/{i}.jpg')
            i += 1
        else: break

    for idx, pth in enumerate(train_paths):
        i = idx + off
        os.symlink('/home/further/TLR/' + pth, f'train/images/{i}.jpg')
        with open(f'train/labels/{i}.txt', 'w') as annot_out:
            if os.path.exists(pth[:-4] + '.txt'):
                with open(pth[:-4] + '.txt', 'r') as annot_in:
                    for box in [ln.strip() for ln in annot_in.readlines()]:
                        box = box.split(',')[1:]  # cls, xyxy
                        x, y, w, h = lisa_xyxy2nxywh(float(box[1]), float(box[2]), float(box[3]), float(box[4]))
                        annot_out.write(f'{label_to_int[simplify_label(box[0])]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')
    
    for idx, pth in enumerate(val_paths[::2]):  # [::5]
        i = idx + voff
        os.symlink('/home/further/TLR/' + pth, f'val/images/{i}.jpg')
        with open(f'val/labels/{i}.txt', 'w') as annot_out:
            if os.path.exists(pth[:-4] + '.txt'):
                with open(pth[:-4] + '.txt', 'r') as annot_in:
                    for box in [ln.strip() for ln in annot_in.readlines()]:
                        box = box.split(',')[1:]  # cls, xyxy
                        x, y, w, h = lisa_xyxy2nxywh(float(box[1]), float(box[2]), float(box[3]), float(box[4]))
                        annot_out.write(f'{label_to_int[simplify_label(box[0])]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')


def val_to_metrics_txt():
    from lisa_dataset import LISA
    int_to_label = {0: 'stop', 1: 'warning', 2: 'go'}
    val_data = LISA(split='val')

    img_res = torch.tensor([1280, 960], dtype=torch.int32)
    for idx in range(len(val_data)):
        img, bboxes, labels = val_data[idx]

        # for ultralytics xywh relative, comment out if not using
        bboxes = bboxes.to(dtype=torch.float32)
        bboxes[:, :2], bboxes[:, 2:] = (bboxes[:, 2:] + bboxes[:, :2]) / 2, bboxes[:, 2:] - bboxes[:, :2]
        bboxes[:, :2] /= img_res
        bboxes[:, 2:] /= img_res
        # bboxes (n, 4)
        # labels (n)
        with open(f'val/labels/{idx}.txt', 'w') as out:  # f'Object-Detection-Metrics/groundtruths/groundtruths/{idx}.txt'
            for i, bbox in enumerate(bboxes):
                # out.write(f'{int_to_label[labels[i].item()]} {int(bbox[0].item())} {int(bbox[1].item())} {int(bbox[2].item())} {int(bbox[3].item())}\n')
                out.write(f'{labels[i].item()} {bbox[0].item():.6f} {bbox[1].item():.6f} {bbox[2].item():.6f} {bbox[3].item():.6f}\n')

def pt_to_yolo(key):
    key = ['yolo'] + key.split('.')[2:]
    return

def retrieve_weights():
    anchors = [[[3, 6],
                [5, 9],
                [8, 13]],
                [[10, 18],
                 [13, 23],
                 [17, 26]]]
    anchors = torch.tensor(anchors, dtype=torch.float32, device='cpu')
    
    yolo = YOLO(anchors, init_resolution=(416, 416), yolo_reduction_factor=32, obj_label_smoothing=0.005, cls_label_smoothing=0.01)
    pt = torch.hub.load('ultralytics/yolov3', 'custom', path='yolov3-tiny.pt')

    # see non matching keys (just layer 20 - model.20.m.0/1...)
    # pt_dict = {'.'.join(['yolo'] + key.split('.')[2:]) : val for key, val in pt.state_dict().items()}
    # yolo_dict = yolo.state_dict()
    # for idx, key in enumerate(pt_dict.keys()):
    #     if pt_dict[key].shape != yolo_dict[key].shape:
    #         print(key, pt_dict[key].shape, yolo_dict[key].shape)
    
    # print(pt.state_dict().keys())
    # print('\n'*3)
    # print({'.'.join(['yolo'] + key.split('.')[2:]) : val for key, val in pt.state_dict().items() if not key.split('.')[3] == '20'}.keys())
    missing, extra = yolo.load_state_dict({'.'.join(['yolo'] + key.split('.')[2:]) : val for key, val in pt.state_dict().items() if not key.split('.')[3] == '20'}, strict=False)
    print(missing, extra)
    torch.save(yolo, 'yolov3-tiny/yolov3-tiny.pt')
    # yolo = torch.load('yolov3-tiny/yolov3-tiny.pt')


def bosch_simplify_label(label):
    if label[:3].lower() == 'red': return 'red'
    if label[:5].lower() == 'green': return 'green'
    if label[:6].lower() == 'yellow': return 'yellow'
    else:
        if label != 'off':
            print('label != off', label)
            assert False
        return label


def bosch_xyxy2nxywh(xmin, ymin, xmax, ymax):
    x, y, w, h = (xmin + xmax) / 2, (ymin + ymax) / 2, (xmax - xmin), (ymax - ymin)
    # resolution = (720, 1280)
    return x / 1280, y / 720, w / 1280, h / 720

def bosch():
    label_to_int = {'red': 0, 'yellow': 1, 'green': 2}

    with open('bosch_test/test.yaml', 'r') as file:
        data = yaml.safe_load(file)
    
    # make txt file with list of all image paths
    with open('bosch_test/img_paths.txt', 'r') as img_files:
        img_paths = [img_file.strip() for img_file in img_files.readlines()]
        img_paths_dict = {'/'.join(pth.split('/')[-1:]) : pth for pth in img_paths}  # [-1:] for test, [-2:] for train

    for annot in data:
        # annot - dict['boxes', 'path']
        annot_path = img_paths_dict['/'.join(annot['path'].split('/')[-1:])][:-4] + '.txt'
        with open(annot_path, 'w') as annot_out:
            for box in annot['boxes']:
                cls = bosch_simplify_label(box['label'])
                if box['occluded'] or cls == 'off': continue
                x, y, w, h = bosch_xyxy2nxywh(box['x_min'], box['y_min'], box['x_max'], box['y_max'])
                annot_out.write(f'{label_to_int[cls]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')
    
    for img_path in img_paths:
        if not os.path.exists(img_path[:-4] + '.txt'):
            print('no annot txt file')
            with open(img_path[:-4] + '.txt', 'w') as annot_out:
                pass

def bosch_to_ultralytics(off, voff):  # offset for the images that are already in train/val folders
    with open('bosch_train/img_paths.txt', 'r') as img_files:
        train_paths = [file.strip() for file in img_files]

    with open('bosch_test/img_paths.txt', 'r') as img_files:
        val_paths = [file.strip() for file in img_files]
    
    for idx, pth in enumerate(train_paths):
        i = idx + off
        os.symlink(pth, f'train/images/{i}.jpg')
        os.symlink(pth[:-4] + '.txt', f'train/labels/{i}.txt')
    
    for idx, pth in enumerate(val_paths[::10]):
        i = idx + voff
        os.symlink(pth, f'val/images/{i}.jpg')
        os.symlink(pth[:-4] + '.txt', f'val/labels/{i}.txt')

def tiff2jpg(pth):
    # Load image from file path, do debayering and shift
    img = cv2.imread(pth, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BAYER_GB2BGR)
    # Images are saved in 12 bit raw -> shift 4 bits
    img = np.right_shift(img, 4)
    img = img.astype(np.uint8)
    return img

def driveu():
    with open('driveu/Bremen.txt', 'r') as images:
        img_paths = [pth.strip() for pth in images.readlines()]

    img_paths.sort()

    # read every some and convert to jpg, then delete the tiff files
    for i in range(0, len(img_paths), 1): # step 2 for everything but bremen
        img = tiff2jpg(img_paths[i])
        try:
            cv2.imwrite(img_paths[i][:-5] + '.jpg', img)
        except Exception as e:
            print(e)
            print('idx:', i)
            return
        
def driveu_simplify_label(label):
    if 'red' in label: return 'red'
    if 'yellow' in label: return 'yellow'
    if 'green' in label: return 'green'
    else:
        print(label)
        assert False

def driveu_include_box(attributes):
    return attributes['direction'] == 'front' and attributes['occlusion'] == 'not_occluded' and (attributes['pictogram'] not in ('pedestrian', 'bicycle')) \
           and attributes['state'] != 'unknown' and attributes['state'] != 'off'

def driveu_xywh2nxywh(label):  # upper left corner, w and h to normalized standard xywh
    # (1024, 2048)
    x, y, w, h = label['x'] + label['w'] / 2, label['y'] + label['h'] / 2, label['w'], label['h']
    return x / 2048, y / 1024, w / 2048, h / 1024
        
def annot_driveu():
    label_to_int = {'red': 0, 'yellow': 1, 'green': 2}

    for json_ in os.listdir('driveu/labels/v2.0'):
        if json_.startswith('DTLD'): continue
        with open('/home/further/TLR/driveu/labels/v2.0/' + json_, 'r') as labels_file:
            labels = json.load(labels_file)

        for img_labels in labels['images']:
            img_path = img_labels['image_path'][2:-5] if img_labels['image_path'].startswith('./') else ('/'.join(img_labels['image_path'].split('/')[4:]))[:-5]
            img_path += '.jpg'
            city = img_path.split('/')[0]
            if os.path.exists(f'/home/further/TLR/driveu/{city}/{img_path}'):
                with open(f'/home/further/TLR/driveu/{city}/{img_path[:-4] + ".txt"}', 'w') as annot_out:
                    for annot in img_labels['labels']:
                        # attribues - direction: front, occlusion: not_occluded, pictogram: not pedestrian or bicycle, state: not unknown
                        if driveu_include_box(annot['attributes']):
                            cls = driveu_simplify_label(annot['attributes']['state'])
                            x, y, w, h = driveu_xywh2nxywh(annot)
                            annot_out.write(f'{label_to_int[cls]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')

    with open('driveu/img_paths.txt', 'r') as image_paths:
        img_paths = [img_path.strip() for img_path in image_paths.readlines()]

    for img_path in img_paths:
        if not os.path.exists(img_path[:-4] + '.txt'):
            print(img_path)
            print('no annot txt file')
            assert False
            with open(img_path[:-4] + '.txt', 'w') as annot_out:
                pass

def driveu_to_ultralytics(off, voff):  # offset for the images that are already in train/val folders
    with open('driveu/img_paths.txt', 'r') as img_files:
        paths = [file.strip() for file in img_files]
        train_paths, val_paths = [], []
        for pth in paths:
            if 'Bremen' in pth: val_paths.append(pth)
            else: train_paths.append(pth)
    
    for idx, pth in enumerate(train_paths):
        i = idx + off
        os.symlink(pth, f'train/images/{i}.jpg')
        os.symlink(pth[:-4] + '.txt', f'train/labels/{i}.txt')
    
    for idx, pth in enumerate(val_paths):
        i = idx + voff
        os.symlink(pth, f'val/images/{i}.jpg')
        os.symlink(pth[:-4] + '.txt', f'val/labels/{i}.txt')

# if idx == 0:
#     fig, axs = plt.subplots(3, 2)
#     data = [pred[..., 0].clone().detach().cpu().sigmoid(), tobj.clone().detach().cpu()]
#     for v in range(3):
#         sns.heatmap(data=data[0][1, v, :, :], vmin=0, vmax=1, ax=axs[v, 0], cmap=sns.color_palette("rocket_r", as_cmap=True))
#         sns.heatmap(data=data[1][1, v, :, :], vmin=0, vmax=1, ax=axs[v, 1], cmap=sns.color_palette("rocket_r", as_cmap=True))
#     plt.savefig('training.png', dpi=100)
#     plt.close()

# new_boxes = np.array([[0, 5, 0, 0],
#                       [0, 3, 0, 0]])
# activate_zones = np.array([0, 0], dtype=np.int32)
# new_cls = np.array([0, 1])
# cls = np.array([-1, -1])
# highest_box = []
# for zidx in np.unique(activate_zones):
#     msk = activate_zones == zidx
#     box_cpy = new_boxes.copy().astype(np.float32)
#     box_cpy[~msk] = np.inf
#     closest_box = box_cpy[:, 1].argmin()  # box that is highest vertically stays
#     highest_box.append(closest_box)
# highest_box = np.array(highest_box, dtype=np.int32)
# cls[activate_zones[highest_box]] = new_cls[highest_box]
# print(cls)

def xywh2xyxy(box):
    wh_ = box[:, 2:] / 2
    xyxy = torch.cat([box[:, :2] - wh_, box[:, :2] + wh_], dim=1)
    return xyxy

def vis_driveu():
    with open('driveu/img_paths.txt', 'r') as image_paths:
        img_paths = [img_path.strip() for img_path in image_paths.readlines()]
    
    print(img_paths[5000])
    img = torch.from_numpy(cv2.imread(img_paths[5000])).permute(2, 0, 1)
    with open(img_paths[5000][:-4] + '.txt', 'r') as annot_in:
        boxes = [[float(coord) for coord in annot.strip().split(' ')[1:]] for annot in annot_in.readlines()]

    boxes = xywh2xyxy(torch.tensor(boxes)).view(-1, 4)
    boxes[:, 0], boxes[:, 2] = boxes[:, 0] * 2048, boxes[:, 2] * 2048
    boxes[:, 1], boxes[:, 3] = boxes[:, 1] * 1024, boxes[:, 3] * 1024
    print(boxes)
    
    img_with_bbox = torchvision.utils.draw_bounding_boxes(img, boxes, colors=(0, 255, 0))
    res = cv2.imwrite('out.jpg', img_with_bbox.permute(1, 2, 0).numpy())
    print(res)

def darken():
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2

    img = cv2.imread('train/images/21599.jpg')  # HWC
    img = np.flip(img, axis=-1)

    img = A.move_tone_curve(img, low_y=0, high_y=0.01)

    img = np.flip(img, axis=-1)
    res = cv2.imwrite('out.jpg', img)

# reset timer if detected again
# count timer if not detected until it hits persistence, then remove
# overlaps_det = iou_grid.max(dim=1) > 0.3
# oot = self.timer >= self.persistence  # out of time
# persist = ~overlaps_det & ~oot  # (n current)

# self.timer = np.concatenate([np.zeros((new_boxes.shape[0],), dtype=np.int32), self.timer[persist] + 1], axis=0)
# self.boxes = np.concatenate([new_boxes, self.boxes[persist]], axis=0)
# self.cls = np.concatenate([new_cls, self.cls[persist]], axis=0)

if __name__ == '__main__':
    # display_img_with_box('daySequence1/daySequence1/frames/daySequence1--00000.jpg')
    # create_annot_files()
    # make_train_test()
    # make_train_val_seq()
    # val_to_metrics_txt()
    # retrieve_weights()
    # bosch()
    # lisa_to_ultralytics()
    # driveu()
    # annot_driveu()

    # driveu_to_ultralytics(0, 0)
    # bosch_to_ultralytics(16918, 591)
    # lisa_to_ultralytics(22011, 1424)

    pass

# train6 is most up to date yolo s
# train7 is 640 res yolo s
# train5 is most up to date yolo m

# find /home/further/TLR/bosch_train -name "*.png" -print > bosch_train/img_paths.txt
# find /home/further/TLR/driveu -name Bremen -prune -o -name '*k0.tiff' -print > driveu/img_paths.txt

# yolo detect train model=/home/further/TLR/runs/detect/train5/weights/last.pt pretrained=False resume=True data=ultralytics/dataset.yaml epochs=14 batch=32 imgsz=416 device=0 optimizer=AdamW cos_lr=True lr0=0.0003 warmup_epochs=0.01