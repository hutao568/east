import math
import random
import cv2
import numpy as np
import torchvision
from PIL import Image


__all__ = ['Compose', 'HeightJitter', 'RandomRotate', 'ColorJitter', 'Noise', 'east_aug']


class Compose(object):
    '''Composes several transforms together.'''

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, polys):
        for t in self.transforms:
            img, polys = t(img, polys)
        return img, polys

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class HeightJitter(object):
    def __init__(self, ratio=0.2):
        self.ratio = ratio

    def __call__(self, img, polys):
        ratio_h = 1 + self.ratio * (np.random.rand() * 2 - 1)
        old_h, old_w, _ = img.shape
        new_h = int(np.around(old_h * ratio_h))
        img = cv2.resize(img, (old_w, new_h))

        new_polys = polys.copy()
        if polys.shape[0] > 0:
            new_polys[:, :, 1] = polys[:, :, 1] * (new_h / old_h)

        return img, new_polys


class RandomRotate(object):
    def __init__(self, angle_range=10):
        self.angle_range = angle_range

    def get_rotate_mat(self, theta):
        return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

    def rotate_polys(self, poly, theta, anchor=None):
        v = poly.T
        if anchor is None:
            anchor = v[:, :1]
        rotate_mat = self.get_rotate_mat(theta)
        res = np.dot(rotate_mat, v - anchor)
        return (res + anchor).T

    def __call__(self, img, polys):
        h, w, _ = img.shape
        center_x = (w - 1) / 2
        center_y = (h - 1) / 2
        angle = self.angle_range * (np.random.rand() * 2 - 1)
        r_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, scale=1) # 逆时针是正方向
        img = cv2.warpAffine(img, r_matrix, (w, h))

        new_polys = np.zeros(polys.shape)
        for i in range(polys.shape[0]):
            new_polys[i, ...] = self.rotate_polys(polys[i],
                                                  -angle / 180 * math.pi,
                                                  np.array([[center_x],[center_y]]))

        return img, new_polys


class ColorJitter(object):
    def __init__(self, p=0.6, brightness=0, contrast=0, saturation=0, hue=0):
        self.p = p
        self.transform = torchvision.transforms.ColorJitter(brightness,
                                                            contrast,
                                                            saturation,
                                                            hue)

    def __call__(self, img, polys):
        if polys.shape[0] == 0:
            # background
            p = self.p
            if np.random.rand() < p:
                img = Image.fromarray(img[:, :, ::-1])
                img = self.transform(img)
                img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        else:
            p = 0.1
            if np.random.rand() < p:
                img = Image.fromarray(img[:, :, ::-1])
                img = self.transform(img)
                img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

        return img, polys


class Noise(object):
    '''add gauss noise'''
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, polys):
        if np.random.rand() < self.p:
            img = img.astype(np.float32)
            mean = random.random() * 0.1
            var = random.random() * 0.01
            img /= 255.
            noise = np.random.normal(mean, var ** 0.5, img.shape)
            img = img + noise
            img = np.clip(img, 0, 1.0)
            img = np.uint8(img * 255)
        return img, polys


east_aug = Compose([HeightJitter(0.2),
                    RandomRotate(10),
                    ColorJitter(0.6, 0.5, 0.5, 0.5, 0.25),
                    #Noise(0.5)
                   ]
            )
