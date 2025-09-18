import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple CNN for image completion.
    Expects inputs: x [B, H, W, C], mask [B, H, W, C]
    """

    def __init__(self, configs):
        super().__init__()
        self.img_size = configs.image_size
        self.in_chans = configs.c_in
        self.hidden_dim = configs.d_model

        self.net = nn.Sequential(
            nn.Conv2d(self.in_chans, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.hidden_dim, self.in_chans, kernel_size=3, padding=1)
        )

    def forward(self, x, mask):
        """
        x: [B, H, W, C]
        mask: [B, H, W, C], 1: visible, 0: missing
        """
        x = x.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)

        x_masked = x * mask
        out = self.net(x_masked)
        out = out.permute(0, 2, 3, 1)
        return out
