import torch
import torch.nn as nn
import os
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from visdom import Visdom
import numpy as np
#import torch.nn.functional as F
import random
import numbers
import torchvision.transforms.functional as F
class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self,x):
        big_img, img, target = x
        if random.random() < self.p:
            return T.functional.hflip(big_img), T.functional.hflip(img), T.functional.hflip(target)
        return big_img, img, target

class RandomVerticalFlip(T.RandomVerticalFlip):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self,x):
        big_img,img,target = x
        if random.random() < self.p:
            return T.functional.vflip(big_img), T.functional.vflip(img), T.functional.vflip(target)
        return big_img, img, target


class RandomCrop(T.RandomCrop):
    def __init__(
        self,
        size=None,
        padding=None,
        pad_if_needed=False,
        fill=0,
        padding_mode='constant'
    ):
        super(RandomCrop, self).__init__(
            size=size,
            padding=padding,
            pad_if_needed=pad_if_needed,
            fill=fill,
            padding_mode=padding_mode
        )
    def __call__(self, x):
        big_img, img, target = x
        size = [int(img.size[1]*0.8),int(img.size[0]*0.8)]
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            target = F.pad(img, self.padding, self.fill, self.padding_mode)

        if self.pad_if_needed and img.size[0] < size[1]:
            img = F.pad(img, (size[1] - img.size[0], 0), self.fill, self.padding_mode)
            target = F.pad(target, (size[1] - target.size[0], 0), self.fill, self.padding_mode)
        if self.pad_if_needed and img.size[1] < size[0]:
            img = F.pad(img, (0, size[0] - img.size[1]), self.fill, self.padding_mode)
            target = F.pad(target, (0, size[0] - target.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, size)

        return T.functional.crop(big_img, i, j , h, w), T.functional.crop(img, i, j , h, w), T.functional.crop(target, i, j, h, w)


class RandomRotation(T.RandomRotation):
    def __init__(self,degrees,resample=False, expand=False, center=None):
        super(RandomRotation, self).__init__(
            degrees,
            resample=resample,
            expand=expand,
            center=center
        )

    def __call__(self,x):
        big_img, img, target = x
        angle = self.get_params(self.degrees)
        return T.functional.rotate(big_img, angle, self.resample, self.expand, self.center), T.functional.rotate(img, angle, self.resample, self.expand, self.center), T.functional.rotate(target, angle, self.resample, self.expand, self.center)


class ColorJitter(T.ColorJitter):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(T.Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(T.Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(T.Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(T.Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = T.Compose(transforms)

        return transform

    def __call__(self, x):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        big_img, img, targets = x
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(big_img), transform(img), targets

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string


class Normalize(T.Normalize):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        big_img, img, target = tensor
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return F.normalize(big_img, self.mean, self.std, self.inplace), F.normalize(img, self.mean, self.std, self.inplace), target


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class ToTensor(T.ToTensor):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        big_img, img, target = pic
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(big_img), F.to_tensor(img), F.to_tensor(target)


    def __repr__(self):
        return self.__class__.__name__ + '()'
