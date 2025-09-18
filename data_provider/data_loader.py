import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
import random
from PIL import Image

# Sentinel-2 standard band order
S2_STANDARD_ORDER = list(range(13))

# EuroSAT-MS band mapping
EUROSAT_MS_BAND_ORDER = {
    'B2': 0, 'B3': 1, 'B4': 2, 'B8': 3,
    'B5': 4, 'B6': 5, 'B7': 6, 'B8A': 7,
    'B11': 8, 'B12': 9, 'B1': 10, 'B9': 11, 'B10': 12
}


def generate_block_mask(h, w, missing_rate, block_size=8):
    """
    Generate a block-wise mask for image inpainting.
    1: visible, 0: missing
    """
    mask = np.ones((h, w), dtype=np.float32)
    num_blocks_h = int(np.ceil(h / block_size))
    num_blocks_w = int(np.ceil(w / block_size))
    total_blocks = num_blocks_h * num_blocks_w
    num_masked = int(total_blocks * missing_rate)

    all_blocks = [(i, j) for i in range(num_blocks_h) for j in range(num_blocks_w)]
    masked_blocks = random.sample(all_blocks, num_masked)
    for (bi, bj) in masked_blocks:
        h_start = bi * block_size
        w_start = bj * block_size
        h_end = min(h_start + block_size, h)
        w_end = min(w_start + block_size, w)
        mask[h_start:h_end, w_start:w_end] = 0.0
    return mask


class Dataset_EuroSAT_MS(Dataset):
    """
    EuroSAT-MS dataset with 13 channels.
    Returns: (x, mask) with shape (H, W, C)
    """
    def __init__(self, configs, image_size=(64, 64), artificially_missing_rate=0.5, flag='train'):
        self.root_dir = configs.data_path
        self.image_size = image_size
        self.missing_rate = artificially_missing_rate
        self.flag = flag

        self.files = []
        for class_dir in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                self.files.extend(glob.glob(os.path.join(class_path, "*.tif")))

        # train / val / test
        self.files.sort()
        n_total = len(self.files)
        n_train = int(0.7 * n_total)
        n_val = int(0.2 * n_total)
        if flag == 'train':
            self.files = self.files[:n_train]
        elif flag == 'val':
            self.files = self.files[n_train:n_train+n_val]
        else:
            self.files = self.files[n_train+n_val:]

        self.channels = S2_STANDARD_ORDER

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        while True:
            path = self.files[idx]
            try:
                with rasterio.open(path) as src:
                    data = src.read().astype(np.float32) / 10000.0
                    data = data[self.channels, :, :]
                if data.shape[1:3] != self.image_size:
                    data = np.stack([np.array(Image.fromarray(d).resize(self.image_size, resample=Image.BILINEAR))
                                     for d in data])
                data = np.transpose(data, (1, 2, 0))  # HWC
            except Exception as e:
                # print(f"Warning: failed to read {path}, skipping. Error: {e}")
                idx = random.randint(0, len(self.files)-1)
                continue

            mask = generate_block_mask(self.image_size[0], self.image_size[1], self.missing_rate)
            mask = np.expand_dims(mask, axis=-1)
            x = data
            return torch.tensor(x, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


class Dataset_EuroSAT_RGB(Dataset):
    """
    EuroSAT-MS RGB-only dataset.
    Channels: B4(R), B3(G), B2(B)
    """
    def __init__(self, configs, image_size=(64, 64), artificially_missing_rate=0.5, flag='train'):
        self.root_dir = configs.data_path
        self.image_size = image_size
        self.missing_rate = artificially_missing_rate
        self.flag = flag

        self.files = []
        for class_dir in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                self.files.extend(glob.glob(os.path.join(class_path, "*.tif")))

        # train / val / test
        self.files.sort()
        n_total = len(self.files)
        n_train = int(0.7 * n_total)
        n_val = int(0.2 * n_total)
        if flag == 'train':
            self.files = self.files[:n_train]
        elif flag == 'val':
            self.files = self.files[n_train:n_train+n_val]
        else:
            self.files = self.files[n_train+n_val:]

        self.channels = [EUROSAT_MS_BAND_ORDER['B4'],
                         EUROSAT_MS_BAND_ORDER['B3'],
                         EUROSAT_MS_BAND_ORDER['B2']]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        while True:
            path = self.files[idx]
            try:
                with rasterio.open(path) as src:
                    data = src.read().astype(np.float32) / 10000.0
                    data = data[self.channels, :, :]
                if data.shape[1:3] != self.image_size:
                    data = np.stack([np.array(Image.fromarray(d).resize(self.image_size, resample=Image.BILINEAR))
                                     for d in data])
                data = np.transpose(data, (1, 2, 0))
            except Exception as e:
                # print(f"Warning: failed to read {path}, skipping. Error: {e}")
                idx = random.randint(0, len(self.files)-1)
                continue

            mask = generate_block_mask(self.image_size[0], self.image_size[1], self.missing_rate)
            mask = np.expand_dims(mask, axis=-1)
            x = data
            return torch.tensor(x, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


class Dataset_EuroSAT_NIR(Dataset):
    """
    EuroSAT-MS NIR-only dataset.
    Channel: B8 (NIR)
    """
    def __init__(self, configs, image_size=(64, 64), artificially_missing_rate=0.5, flag='train'):
        self.root_dir = configs.data_path
        self.image_size = image_size
        self.missing_rate = artificially_missing_rate
        self.flag = flag

        self.files = []
        for class_dir in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_dir)
            if os.path.isdir(class_path):
                self.files.extend(glob.glob(os.path.join(class_path, "*.tif")))

        # train / val / test
        self.files.sort()
        n_total = len(self.files)
        n_train = int(0.7 * n_total)
        n_val = int(0.2 * n_total)
        if flag == 'train':
            self.files = self.files[:n_train]
        elif flag == 'val':
            self.files = self.files[n_train:n_train+n_val]
        else:
            self.files = self.files[n_train+n_val:]

        self.channels = [EUROSAT_MS_BAND_ORDER['B8']]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        while True:
            path = self.files[idx]
            try:
                with rasterio.open(path) as src:
                    data = src.read().astype(np.float32) / 10000.0
                    data = data[self.channels, :, :]
                if data.shape[1:3] != self.image_size:
                    data = np.stack([np.array(Image.fromarray(d).resize(self.image_size, resample=Image.BILINEAR))
                                     for d in data])
                data = np.transpose(data, (1, 2, 0))
            except Exception as e:
                # print(f"Warning: failed to read {path}, skipping. Error: {e}")
                idx = random.randint(0, len(self.files)-1)
                continue

            mask = generate_block_mask(self.image_size[0], self.image_size[1], self.missing_rate)
            mask = np.expand_dims(mask, axis=-1)
            x = data
            return torch.tensor(x, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)
