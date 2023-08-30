# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset


class PFB(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=19,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(PFB, self).__init__(ignore_label, base_size,
                                         crop_size, downsample_rate, scale_factor, mean, std, )

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip

        self.img_list = [line.strip().split() for line in open(root + list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

        self.label_mapping = {0: ignore_label,
                              1: ignore_label, 2: 0,
                              3: 1, 4: 2,
                              5: ignore_label, 6: 3,
                              7: 4, 8: 5, 9: 6,
                              10: 7, 11: 8, 12: 9,
                              13: 10, 14: 11, 15: 12,
                              16: 13, 17: 14, 18: 15,
                              19: 16, 20: 17, 21: ignore_label, 22: ignore_label, 23: 18, 24: 19,
                              25: 20, 26: 21, 27: 22, 28: ignore_label,
                              29: ignore_label, 30: ignore_label,
                              31: ignore_label}

        # self.label_mapping = {
        #     0:  {'name': 'unlabeled', 'color': (0, 0, 0), 'train_id': 255},
        #     1:  {'name': 'ambiguous', 'color': (111, 74, 0), 'train_id': 255},
        #     2:  {'name': 'sky', 'color': (70, 130, 180), 'train_id': 0},
        #     3:  {'name': 'road', 'color': (128, 64, 128), 'train_id': 1},
        #     4:  {'name': 'sidewalk', 'color': (244, 35, 232), 'train_id': 2},
        #     5:  {'name': 'rail track', 'color': (230, 150, 140), 'train_id': 255},
        #     6:  {'name': 'terrain', 'color': (152, 251, 152), 'train_id': 3},
        #     7:  {'name': 'tree', 'color': (87, 182, 35), 'train_id': 4},
        #     8:  {'name': 'vegetation', 'color': (35, 142, 35), 'train_id': 5},
        #     9:  {'name': 'building', 'color': (70, 70, 70), 'train_id': 6},
        #     10: {'name': 'infrastructure', 'color': (153, 153, 153), 'train_id': 7},
        #     11: {'name': 'fence', 'color': (190, 153, 153), 'train_id': 8},
        #     12: {'name': 'billboard', 'color': (150, 20, 20), 'train_id': 9},
        #     13: {'name': 'trafficlight', 'color': (250, 170, 30), 'train_id': 10},
        #     14: {'name': 'traffic sign', 'color': (220, 220, 0), 'train_id': 11},
        #     15: {'name': 'mobile barrier', 'color': (180, 180, 100), 'train_id': 12},
        #     16: {'name': 'fire hydrant', 'color': (173, 153, 153), 'train_id': 13},
        #     17: {'name': 'chair', 'color': (168, 153, 153), 'train_id': 14},
        #     18: {'name': 'trash', 'color': (81, 0, 21), 'train_id': 15},
        #     19: {'name': 'trashcan', 'color': (81, 0, 81), 'train_id': 16},
        #     20: {'name': 'person', 'color': (220, 20, 60), 'train_id': 17},
        #     21: {'name': 'animal', 'color': (255, 0, 0), 'train_id': 255},
        #     22: {'name': 'bicycle', 'color': (119, 11, 32), 'train_id': 255},
        #     23: {'name': 'motorcycle', 'color': (0, 0, 230), 'train_id': 18},
        #     24: {'name': 'car', 'color': (0, 0, 142), 'train_id': 19},
        #     25: {'name': 'van', 'color': (0, 80, 100), 'train_id': 20},
        #     26: {'name': 'bus', 'color': (0, 60, 100), 'train_id': 21},
        #     27: {'name': 'truck', 'color': (0, 0, 70), 'train_id': 22},
        #     28: {'name': 'trailer', 'color': (0, 0, 90), 'train_id': 255},
        #     29: {'name': 'train', 'color': (0, 80, 100), 'train_id': 255},
        #     30: {'name': 'plane', 'color': (0, 100, 100), 'train_id': 255},
        #     31: {'name': 'boat', 'color': (50, 0, 90), 'train_id': 255},
        # }

        self.class_weights = torch.FloatTensor([1.0, 1.0, 1.0, 1.0,
                                                1.0, 1.0, 1.0, 1.0,
                                                1.0, 1.0, 1.0, 1.0,
                                                1.0, 1.0, 1.0, 1.0,
                                                1.0, 1.0, 1.0, 1.0,
                                                1.0, 1.0, 1.0]).cuda()

    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files

    def convert_label(self, label, inverse=False):
        if inverse:
            label_image_gray = label.copy()
        else:
            label_image_gray = label.copy()  # np.ones(label.shape[:2]) * 255
            # for key in self.label_mapping.keys():
            #     indices = np.argwhere(np.all(label == self.label_mapping[key]['color'], axis=-1))
            #     rr, cc = indices[:, 0], indices[:, 1]
            #     label_image_gray[rr, cc] = self.label_mapping[key]['train_id']
        return label_image_gray

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        # image = cv2.imread(os.path.join(self.root,'cityscapes',item["img"]),
        #                    cv2.IMREAD_COLOR)
        # image = cv2.imread(os.path.join(self.root, item["img"]),
        #                    cv2.IMREAD_COLOR)
        image = Image.open(os.path.join(self.root, item["img"])).convert('RGB')
        image = np.array(image)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        # label = cv2.imread(os.path.join(self.root,'cityscapes',item["label"]),
        #                    cv2.IMREAD_GRAYSCALE)
        # labelel = cv2.imread(os.path.join(self.root, item["label"]),
        #                    cv2.IMREAD_GRAYSCALE)
        label = Image.open(os.path.join(self.root, item["label"])).convert('RGB')
        label = np.array(label)
        label = self.convert_label(label)

        image, label = self.gen_sample(image, label,
                                       self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name

    def multi_scale_inference(self, config, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 1.0)
        stride_w = np.int(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                  ori_height, ori_width]).cuda()
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]

            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(config, model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                     new_h, new_w]).cuda()
                count = torch.zeros([1, 1, new_h, new_w]).cuda()

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]

            preds = F.interpolate(
                preds, (ori_height, ori_width),
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))



