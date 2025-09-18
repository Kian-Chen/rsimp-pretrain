import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple Masked Autoencoder (MAE) for image completion.
    Expects inputs: x [B, H, W, C], mask [B, H, W, C]
    """

    def __init__(self, configs):
        super().__init__()
        self.img_size = configs.image_size
        self.in_chans = configs.c_in
        self.embed_dim = configs.d_model

        # Encoder: simple conv layers
        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_chans, self.embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder: reconstruct original image
        self.decoder = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.embed_dim, self.in_chans, kernel_size=3, padding=1)
        )

    def forward(self, x, mask):
        """
        x: [B, H, W, C]
        mask: [B, H, W, C], 1: visible, 0: missing
        """
        # Convert to [B, C, H, W] for conv
        x = x.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)

        # Encode masked input
        x_masked = x * mask
        z = self.encoder(x_masked)

        # Decode to reconstruct
        recon = self.decoder(z)

        # Return reconstructed image in [B, H, W, C]
        recon = recon.permute(0, 2, 3, 1)
        return recon
