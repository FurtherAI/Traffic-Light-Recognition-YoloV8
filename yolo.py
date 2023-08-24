'''
Source for Yolov3 model implementation
@software{Jocher_by_Ultralytics_2020,
author = {Jocher, Glenn},
doi = {10.5281/zenodo.3908559},
license = {AGPL-3.0},
month = may,
title = {{ by Ultralytics}},
url = {https://github.com/ultralytics/yolov5},
version = {7.0},
year = {2020}
}
'''

import argparse
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path
import torch
from torch import nn
import math

from layers import Conv, Concat, Bottleneck, BottleneckCSP

class Detect(nn.Module):
    # YOLOv3 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=3, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            # if not self.training:  # inference
            #     if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
            #         self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            #     xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
            #     xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
            #     wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
            #     y = torch.cat((xy, wh, conf), 4)
            #     z.append(y.view(bs, self.na * nx * ny, self.no))

        # return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
        return x

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class BaseModel(nn.Module):
    # YOLOv3 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # if profile:
            #     self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
            # if visualize:
            #     feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    # def _profile_one_layer(self, m, x, dt):
    #     c = m == self.model[-1]  # is final layer, copy input as inplace fix
    #     o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
    #     t = time_sync()
    #     for _ in range(10):
    #         m(x.copy() if c else x)
    #     dt.append((time_sync() - t) * 100)
    #     if m == self.model[0]:
    #         LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
    #     LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
    #     if c:
    #         LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")


class DetectionModel(BaseModel):
    # YOLOv3 detection model
    def __init__(self, cfg='yolov3-tiny.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc  # override yaml value
        # if anchors is not None:
        #     self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        self.inplace = self.yaml.get('inplace', True)

        # Init weights, biases
        self.initialize_weights()

    def forward(self, x, augment=False, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def initialize_weights(self):
        for m in self.modules():
            t = type(m)
            if t is nn.Conv2d:
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                pass
            elif t is nn.BatchNorm2d:
                m.eps = 1e-3
                m.momentum = 0.03

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain  'Model' class for backwards compatibility

def make_divisible(x, divisor):
    # Returns nearest x divisible by divisor
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())  # to int
    return math.ceil(x / divisor) * divisor

def parse_model(d, ch):  # model_dict, input_channels(3)
    # Parse a  model.yaml dictionary
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation')
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            # with contextlib.suppress(NameError):
            args[j] = eval(a) if (isinstance(a, str) and a != 'nearest') else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
                Conv, Bottleneck}:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)



