import random
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

class Normalize(object):
    '''
    Normalizes the color range to [0,1] and converts the color image to grayscale
    '''
    def __init__(self, color=False):
        self.color = color

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        if not self.color:
            image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        image_copy = image_copy / 255.0

        return {"image": image_copy, "keypoints": key_pts}

class Rescale(object):
    '''
    Rescale the image in a sample to a given size while maintaining aspect ratio.
   
    Optionally, flip the image if it's wider than it is tall.

    Inputs:
    output_size (tuple or int): Desired output size. 
    '''

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']
        h, w = image.shape[:2]
        
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
                image = cv2.flip(image, 1) 

                key_pts = key_pts.copy()
                for i in range(0, len(key_pts), 2):
                    key_pts[i] = w - key_pts[i]  
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))

        scale_x, scale_y = new_w / w, new_h / h
        scaled_key_pts = np.array([key_pts[0]*scale_x,key_pts[1]*scale_x,key_pts[2]*scale_y,key_pts[3]*scale_y])

        return {'image': img, 'keypoints': scaled_key_pts}

class ToTensor(object):
    '''
    Convert ndarrays in sample to Tensors.
    '''

    def __call__(self, sample):
        image, key_pts = sample["image"], sample["keypoints"]

        if len(image.shape) == 2:
            image = image.reshape(image.shape[0], image.shape[1], 1)

        image = image.transpose((2, 0, 1))
        return {
            "image": torch.from_numpy(image),
            "keypoints": torch.from_numpy(key_pts),
        }
