from ultralytics import YOLO
import torch
import numpy as np
import cv2
from jetson_utils import cudaFromNumpy
from jetson.utils import videoOutput, videoSource

cls_to_color = {0: 'red', 1: 'yellow', 2: 'green'}

class PersistentDisplay():
    def __init__(self, n_zones=7, img_res=(480, 800), persistence=30):
        self.zones = np.zeros((n_zones,), dtype=bool)
        self.cls = np.full((n_zones,), -1, dtype=np.int32)
        self.bins = np.linspace(0, img_res[1], num=n_zones + 1)
        self.timer = np.zeros((n_zones,), dtype=np.int32)

        self.h = img_res[0]
        self.w = img_res[1]
        self.persistence = persistence  # num updates to hold box
        self.cls_to_color = {-1: (0, 0, 0), 0: (0, 0, 255), 1: (0, 234, 255), 2: (0, 255, 0)}

    def xyxy2xywh(self, boxes):
        wh = boxes[..., 2:4] - boxes[..., :2]
        xy = (boxes[..., 2:4] + boxes[..., :2]) / 2
        mins = np.array([0, 0, 0, 0])
        maxs = np.array([self.w, self.h, self.w, self.h])
        return np.clip(np.concatenate([xy, wh], axis=-1), a_min=mins, a_max=maxs)

    def update(self, new_boxes, new_cls):
        new_boxes = self.xyxy2xywh(new_boxes.numpy())
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

        self.cls[activate_zones[highest_box]] = new_cls[highest_box]
        self.timer[activate_zones] = 0

        deactivate_zones[activate_zones] = False
        self.zones[deactivate_zones] = False
        self.cls[deactivate_zones] = -1

        # turn into image
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        for i in range(self.zones.shape[0]):
            cv2.rectangle(img, (int(self.bins[i]), 0), (int(self.bins[i + 1]), self.h), color=self.cls_to_color[self.cls[i]], thickness=cv2.FILLED)

        return img


camera = videoSource('csi://0')
display = videoOutput('display://0')

model = YOLO('runs/detect/yolo_s_640/weights/best.pt')
print('model should be on cuda', model.device)

h, w = 480, 800
fps = 30
persistent_display = PersistentDisplay(n_zones=9, img_res=(h, w), persistence=int(0.8 * fps))  # (960, 1280)

while True:
    camera_img = camera.Capture()
    if camera_img is None: continue

    out = model.track(torch.as_tensor(camera_img, device='cuda'), persist=True) # [0]  # list of len batch
    img = torch.from_numpy(out.orig_img).permute(2, 0, 1).flip(0)

    new_boxes = out.boxes.xyxy.cpu()
    cls = out.boxes.cls.cpu()
    update_display = persistent_display.update(new_boxes, cls)

    display.Render(cudaFromNumpy(update_display))
