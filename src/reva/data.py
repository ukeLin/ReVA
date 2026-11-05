import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .protocol import pairs

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class SegmentationDataset(Dataset):
    def __init__(self, root, split, size=352):
        self.pairs = pairs(root, split)
        self.training = split == "train"
        self.size = size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB").resize((self.size, self.size), Image.Resampling.BILINEAR)
            image = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255)
        with Image.open(mask_path) as mask:
            mask = mask.convert("RGB")
            if self.training:
                mask = mask.resize((self.size, self.size), Image.Resampling.NEAREST)
            mask = torch.from_numpy((np.array(mask).max(-1) > 127).astype(np.float32))[None]
        if self.training:
            for axis in (-1, -2):
                if random.random() < 0.5:
                    image, mask = image.flip(axis), mask.flip(axis)
            turns = int(torch.randint(4, ()))
            image, mask = image.rot90(turns, (-2, -1)), mask.rot90(turns, (-2, -1))
        return (image - MEAN) / STD, mask, image_path.name
