from torch.utils.data import Dataset

from PIL import Image
import os
import numpy as np
import torch
from copy import deepcopy
from typing import Optional, Tuple
class BaseInstanceDataset(Dataset):
    def __init__(self):
        assert False, print("Unimplement Dataset.")

    def __getitem__(self, item):
        pass

class MyDataset_bbox2Seg(Dataset):
    def __init__(self, anno, root, transform, target_transform):
        super(MyDataset_bbox2Seg).__init__()
        self.anno = anno
        self.root = root
        from pycocotools.coco import COCO
        self.coco = COCO(self.anno)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ids)

    def get_image(self, ids):
        img_info = self.coco.loadImgs(ids=[ids])[0]
        image  = Image.open(os.path.join(self.root, img_info['file_name'])).convert('RGB')
        return image

    def get_mask(self, ids):
        anno_id = self.coco.getAnnIds(imgIds=[ids])
        anns = self.coco.loadAnns(ids=anno_id)
        if not anns:
            mask = None
        else:
            mask = np.array([ann['bbox'] for ann in anns]) #xywh
        return mask

    def get_preprocess_shape(self,oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        if neww < 640:
            neww = 640  # wh padding
        padw = long_side_length-neww
        return (newh, neww), padw

    def __getitem__(self, item):
        ids = self.ids[item]
        image = self.get_image(ids)
        mask = self.get_mask(ids)
        width, height = image.size
        if self.transform is not None:
            image = self.transform(image)
            boxes = deepcopy(mask)
            if mask is None:
                boxes = mask
            else:
                boxes[..., 2] = boxes[..., 0] + boxes[..., 2]
                boxes[..., 3] = boxes[..., 1] + boxes[..., 3]
                boxes = self.apply_coords(boxes.reshape(-1, 2, 2), (width, height)).reshape(-1,4)
        return image, boxes

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_w, old_h = original_size
        (new_w, new_h), padw = self.get_preprocess_shape(
            original_size[0], original_size[1], self.transform.transforms[0].target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h) + padw//2

        return coords


def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for i, (img, box) in enumerate(batch):
        images.append(img)
        if box is None:
            continue
        box = torch.tensor(box)
        lb = torch.cat((torch.full((box.size(0), 1), i), box), dim=1)
        bboxes.append(lb)

    images = torch.stack(images)
    bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    return images, bboxes