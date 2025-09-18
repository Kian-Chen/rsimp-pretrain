import numpy as np
import torch
import math

def MAE(pred, true, mask=None):
    """Mean Absolute Error"""
    if mask is not None:
        pred, true = pred[mask], true[mask]
    return np.mean(np.abs(pred - true))

def MSE(pred, true, mask=None):
    """Mean Squared Error"""
    if mask is not None:
        pred, true = pred[mask], true[mask]
    return np.mean((pred - true) ** 2)

def RMSE(pred, true, mask=None):
    """Root Mean Squared Error"""
    return np.sqrt(MSE(pred, true, mask))

def PSNR(pred, true, max_val=255.0, mask=None):
    """Peak Signal-to-Noise Ratio"""
    mse = MSE(pred, true, mask)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_val / np.sqrt(mse))

def SSIM(pred, true, mask=None, max_val=255.0, eps=1e-8):
    """
    SSIM for image completion
    pred, true shape: (B, H, W, C)
    mask: same shape as true, 1 for valid region, 0 for missing
    """
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(true, np.ndarray):
        true = torch.from_numpy(true)
    if mask is not None and isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    pred = pred.permute(0, 3, 1, 2).float()  # (B, C, H, W)
    true = true.permute(0, 3, 1, 2).float()

    if mask is not None:
        mask = mask.permute(0, 3, 1, 2).float()
        mask_sum = mask.sum(dim=[2, 3], keepdim=True) + eps
        mu_pred = (pred * mask).sum(dim=[2, 3], keepdim=True) / mask_sum
        mu_true = (true * mask).sum(dim=[2, 3], keepdim=True) / mask_sum
    else:
        mu_pred = pred.mean(dim=[2, 3], keepdim=True)
        mu_true = true.mean(dim=[2, 3], keepdim=True)

    sigma_pred = ((pred - mu_pred) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma_true = ((true - mu_true) ** 2).mean(dim=[2, 3], keepdim=True)
    sigma_xy = ((pred - mu_pred) * (true - mu_true)).mean(dim=[2, 3], keepdim=True)

    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2

    ssim = ((2 * mu_pred * mu_true + C1) * (2 * sigma_xy + C2)) / \
           ((mu_pred ** 2 + mu_true ** 2 + C1) * (sigma_pred + sigma_true + C2))

    return ssim.mean().item()

def R2(pred, true, mask=None):
    """Coefficient of determination (R² score)"""
    if mask is not None:
        pred, true = pred[mask], true[mask]
    true_mean = np.mean(true)
    numerator = np.sum((true - pred) ** 2)
    denominator = np.sum((true - true_mean) ** 2)
    return 1 - numerator / denominator if denominator != 0 else 0.0

def metric(pred, true, mask=None):
    """
    Return a tuple of metrics for completion task
    pred, true, mask: numpy arrays, shape (B, H, W, C)
    """
    mae = MAE(pred, true, mask)
    mse = MSE(pred, true, mask)
    rmse = RMSE(pred, true, mask)
    psnr = PSNR(pred, true, 255.0, mask)
    ssim = SSIM(pred, true, mask)
    r2 = R2(pred, true, mask)
    return mae, mse, rmse, psnr, ssim, r2
