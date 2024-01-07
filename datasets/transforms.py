import copy

import torchvision.transforms as T
from omegaconf.dictconfig import DictConfig
import torch.nn as nn
import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

from copy import deepcopy
from typing import Tuple

class ResizeLongSide:
    def __init__(self, size: int) -> None:
        self.target_length = size

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, box: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = deepcopy(box)
        boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
        boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        if neww < 640:
            neww = 640   # wh padding
        padw = long_side_length-neww
        return (newh, neww), padw

    def __call__(self, label):
        return self.apply_boxes(label)




class ResizeLongestSideImg(ResizeLongSide):
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size, padw = self.get_preprocess_shape(image.size[0], image.size[1], self.target_length)
        image = np.array(resize(image, target_size))
        image = np.pad(image, ((0, 0), (padw//2, padw//2), (0, 0)))
        return image

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[2], image.shape[3], self.target_length)
        image = F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )
        image = F.pad(image, (padh//2, padh//2), "constant", 114/255)
        return image




    def __call__(self, img):
        return self.apply_image(img)






AVIAL_TRANSFORM = {'resize': T.Resize, 'to_tensor': T.ToTensor, 'img_resize': ResizeLongestSideImg,'label_resize': ResizeLongSide}


def get_transforms(transforms: DictConfig):
    T_list = []
    if transforms is None:
        return None
    for t_name in transforms.keys():
        assert t_name in AVIAL_TRANSFORM, "{T_name} is not supported transform, please implement it and add it to " \
                                          "AVIAL_TRANSFORM first.".format(T_name=t_name)
        if transforms[t_name].params is not None:
            T_list.append(AVIAL_TRANSFORM[t_name](**transforms[t_name].params))
        else:
            T_list.append(AVIAL_TRANSFORM[t_name]())
    return T.Compose(T_list)


class CustomTransform(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass

