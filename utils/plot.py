import os
import matplotlib.pyplot as plt
import torch
import numpy as np

import matplotlib.pyplot as plt
import os
import torch
import numpy as np

def visualize_images(original, reconstructed, titles=None, save_path=None):
    """
    Visualize original and reconstructed images side by side.
    original, reconstructed: torch.Tensor or numpy array, shape (B, H, W, C) or (B, C, H, W)
    titles: optional list of titles
    save_path: where to save the figure
    """
    def to_hwc(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if x.ndim == 4 and x.shape[1] <= 4:  # (B, C, H, W)
            x = x.transpose(0, 2, 3, 1)
        return x

    def normalize_uint8(img):
        # img_min, img_max = img.min(), img.max()
        # if img_max > img_min:
        #     img = (img - img_min) / (img_max - img_min) * 255.0
        # else:
        #     img = np.zeros_like(img)
        img = np.clip(img*255.0, 0, 255)
        return img.astype('uint8')

    orig_imgs = to_hwc(original)
    rec_imgs  = to_hwc(reconstructed)
    num_images = orig_imgs.shape[0]

    plt.figure(figsize=(4 * num_images, 4))
    for i in range(num_images):
        # 原图
        plt.subplot(2, num_images, i + 1)
        plt.imshow(normalize_uint8(orig_imgs[i]))
        plt.axis('off')
        if titles:
            plt.title(titles[i] + ' orig')

        # 重建图
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(normalize_uint8(rec_imgs[i]))
        plt.axis('off')
        if titles:
            plt.title(titles[i] + ' rec')

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()