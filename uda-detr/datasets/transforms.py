# ------------------------------------------------------------------------
# Modified by Wei-Jie Huang
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Transforms and data augmentation for both image + bbox.
"""
import random

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

import numpy as np
# 高斯噪声 AddGaussianNoise(mean=0., std=0.1),
import torch
from PIL import Image, ImageFilter
from torchvision import transforms
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=20.):
        self.mean = mean
        self.std = std

    def __call__(self, img, target):
        # 打印调试信息
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        # print(f"Image size: {img.size()}")
        # print(f"Image type: {type(img)}")
        noise = torch.randn(img.size()) * self.std + self.mean
        noisy_img = img + noise
        noisy_img = transforms.ToPILImage()(noisy_img)
        return noisy_img, target
# 明暗变化 AdjustBrightness(brightness_factor=1.5)
class AdjustBrightness(object):
    def __init__(self, brightness_factor = 1.5):
        self.brightness_factor = brightness_factor

    def __call__(self, img, target):
        adjusted_img = F.adjust_brightness(img, self.brightness_factor)
        return adjusted_img, target
# 对比度变化  AdjustContrast(contrast_factor=1.5),
class AdjustContrast(object):
    def __init__(self, contrast_factor = 1.5):
        self.contrast_factor = contrast_factor

    def __call__(self, img, target):
        adjusted_img = F.adjust_contrast(img, self.contrast_factor)
        return adjusted_img, target
# 饱和度变化

class AdjustSaturation(object):
    def __init__(self, saturation_factor = 1.5):
        self.saturation_factor = saturation_factor

    def __call__(self, img, target):
        adjusted_img = F.adjust_saturation(img, self.saturation_factor)
        return adjusted_img, target
# 色调变化 AdjustHue(hue_factor=0.1)

class AdjustHue(object):
    def __init__(self, hue_factor = 0.1):
        self.hue_factor = hue_factor

    def __call__(self, img, target):
        adjusted_img = F.adjust_hue(img, self.hue_factor)
        return adjusted_img, target

# 随机旋转 RandomRotation(degrees=30),
class RandomRotation(object):
    def __init__(self, degrees = 30):
        self.degrees = degrees

    def __call__(self, img, target):
        angle = random.uniform(-self.degrees, self.degrees)
        rotated_img = F.rotate(img, angle)
        return rotated_img, target


class FixedSizeResize(object):
    def __init__(self, target_size):
        """
        初始化 FixedSizeResize 类

        参数:
        - target_size: 目标大小，格式为 (width, height)
        """
        self.target_size = target_size

    def __call__(self, img, target=None):
        """
        将图像和标注信息调整为指定大小

        参数:
        - img: PIL.Image 对象
        - target: 包含标注信息的字典，例如 {'boxes': 边界框, 'labels': 标签}

        返回:
        - resized_image: 调整大小后的 PIL.Image 对象
        - resized_target: 调整大小后的标注信息字典
        """
        # 调整图像大小
        resized_image = F.resize(img, self.target_size)

        # 如果 target 为 None，则直接返回调整大小后的图像
        if target is None:
            return resized_image, None

        # 获取调整前后的尺寸比例
        orig_width, orig_height = img.size
        new_width, new_height = self.target_size
        width_scale = new_width / orig_width
        height_scale = new_height / orig_height

        # 调整标注信息
        resized_target = target.copy()
        if 'boxes' in resized_target:
            boxes = resized_target['boxes']
            # 调整边界框坐标
            scaled_boxes = boxes * torch.tensor([width_scale, height_scale, width_scale, height_scale])
            resized_target['boxes'] = scaled_boxes

        if 'area' in resized_target:
            area = resized_target['area']
            # 调整面积
            scaled_area = area * (width_scale * height_scale)
            resized_target['area'] = scaled_area

        if 'masks' in resized_target:
            masks = resized_target['masks']
            # 调整掩码大小
            resized_masks = F.resize(masks.unsqueeze(1).float(), [new_height, new_width], interpolation=Image.NEAREST).squeeze(1)
            resized_target['masks'] = resized_masks

        return resized_image, resized_target

class RandomApplyImgAnno(T.RandomApply):

    def __init__(self, transforms, p=0.5):
        super(RandomApplyImgAnno, self).__init__(transforms, p)

    def forward(self, image, annotation=None):
        if self.p < torch.rand(1):
            return image, annotation
        for t in self.transforms:
            image, annotation = t(image, annotation)
        return image, annotation


class ColorJitterImgAnno(T.ColorJitter):
    """
    Color jitter, keep annotation
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(ColorJitterImgAnno, self).__init__(brightness, contrast, saturation, hue)

    def forward(self, image, annotation=None):
        return super(ColorJitterImgAnno, self).forward(image), annotation


class RandomGrayScaleImgAnno(T.RandomGrayscale):
    """
    Random grayscale, keep annotation
    """
    def __init__(self, p=0.1):
        super(RandomGrayScaleImgAnno, self).__init__(p)

    def forward(self, image, annotation=None):
        return super(RandomGrayScaleImgAnno, self).forward(image), annotation


class GaussianBlurImgAnno:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as described in SimCLR.
    """
    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [0.1, 2.0]
        self.sigma = sigma

    def __call__(self, image, annotation=None):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        image = image.filter(ImageFilter.GaussianBlur(radius=sigma))
        return image, annotation
