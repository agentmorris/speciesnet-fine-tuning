#%% Header

"""
dataset.py

A PyTorch Dataset that produces classifier crops on the fly: it reads the full
image, crops to the MegaDetector box (matching SpeciesNet's always_crop
behavior), resizes to the model's input size, and normalizes. Cropping on the
fly avoids duplicating tens of gigabytes of cropped images on disk.
"""

#%% Imports and constants

import os

import torch
import torchvision.transforms as T
from PIL import Image

from model import IMG_SIZE, NORM_MEAN, NORM_STD


#%% Support functions

def crop_resize(im, bbox, img_size=IMG_SIZE):
    """
    Crop a PIL image to a normalized MegaDetector bbox [x, y, w, h] and resize.

    Falls back to the whole image if the box is degenerate. Used by both training
    and inference so the preprocessing is identical.
    """

    width, height = im.size
    x, y, w, h = bbox
    left = max(0, int(round(x * width)))
    top = max(0, int(round(y * height)))
    right = min(width, int(round((x + w) * width)))
    bottom = min(height, int(round((y + h) * height)))
    crop = im if (right <= left or bottom <= top) else im.crop((left, top, right, bottom))
    return crop.resize((img_size, img_size), Image.BICUBIC)


def build_transforms(img_size=IMG_SIZE, train=True, mean=NORM_MEAN, std=NORM_STD):
    """
    Transforms applied to the already-cropped, already-resized PIL image.
    """

    if train:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
            T.ToTensor(),
            T.Normalize(mean, std),
        ])
    return T.Compose([T.ToTensor(), T.Normalize(mean, std)])


class CropDataset(torch.utils.data.Dataset):
    """
    Dataset of MegaDetector crops, labeled by their image's category.
    """

    def __init__(self, instances, class_to_idx, image_root, img_size=IMG_SIZE,
                 train=True, mean=NORM_MEAN, std=NORM_STD):
        self.instances = instances
        self.class_to_idx = class_to_idx
        self.image_root = image_root
        self.img_size = img_size
        self.transform = build_transforms(img_size, train, mean, std)

    def __len__(self):
        return len(self.instances)

    def _resolve(self, filename):
        return filename if os.path.isabs(filename) else os.path.join(self.image_root, filename)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        label = self.class_to_idx[inst.category]
        try:
            with Image.open(self._resolve(inst.filename)) as im:
                im = im.convert("RGB")
                crop = crop_resize(im, inst.bbox, self.img_size)
                tensor = self.transform(crop)
        except Exception:
            # Corrupt or unreadable image: return a black crop so training continues
            tensor = torch.zeros(3, self.img_size, self.img_size)
        return tensor, label
