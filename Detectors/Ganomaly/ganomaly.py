import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# utility: DCGAN-style initialisation
# ----------------------------------------------------------------------
def _weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class View(nn.Module):
    """Reshape inside nn.Sequential."""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


# ----------------------------------------------------------------------
# Decoder  — latent ➜ 160 × 320 image
# ----------------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, n_channels: int = 3, latent_space_dimension: int = 100):
        super().__init__()

        self.net = nn.Sequential(
            # project & reshape to (N, 256, 10, 20)
            nn.Linear(latent_space_dimension, 1024),
            nn.Linear(1024, 256 * 10 * 20),
            nn.BatchNorm1d(256 * 10 * 20, momentum=0.1, eps=1e-5),
            nn.LeakyReLU(0.2, inplace=True),
            View((-1, 256, 10, 20)),

            # 10×20 → 20×40
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 20×40 → 40×80
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # 40×80 → 80×160
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # 80×160 → 160×320
            nn.ConvTranspose2d(32, n_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

        self.apply(_weights_init)

    def forward(self, z):
        return self.net(z)


# ----------------------------------------------------------------------
# Encoder  — 160 × 320 image ➜ latent vector
# ----------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, n_channels: int = 3, latent_space_dimension: int = 100):
        super().__init__()

        self.conv = nn.Sequential(
            # 160×320 → 80×160
            nn.Conv2d(n_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 80×160 → 40×80
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 40×80 → 20×40
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 20×40 → 10×20
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 10×20 → 1×1  (squeeze spatial dims)
            nn.Conv2d(512, latent_space_dimension,
                      kernel_size=(10, 20), stride=1, padding=0, bias=False),
        )

        self.apply(_weights_init)

    def forward(self, x):
        z = self.conv(x)          # (N, latent_dim, 1, 1)
        return z.view(z.size(0), -1)


# ----------------------------------------------------------------------
# Discriminator  — 160 × 320 image ➜ real/fake score + features
# ----------------------------------------------------------------------
class Discriminator(nn.Module):
    """
    Returns
    -------
    out   : (N, 1)   Raw authenticity logits
    feats : (N, F)   Flattened backbone features (useful for feature-matching loss)
    """
    def __init__(self, n_channels: int = 3):
        super().__init__()

        self.backbone = nn.Sequential(
            # 160×320 → 80×160
            nn.Conv2d(n_channels, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 80×160 → 40×80
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 40×80 → 20×40
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 20×40 → 10×20
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # feature map is (N, 256, 10, 20) ⇒ 256×10×20 = 51 200 dims
        self.out_head = nn.Linear(256 * 10 * 20, 1, bias=False)

        self.apply(_weights_init)

    def forward(self, x):
        feats = self.backbone(x)             # (N, 256, 10, 20)
        feats_flat = feats.flatten(1)        # (N, 51 200)
        out = self.out_head(feats_flat)      # (N, 1)
        return out, feats_flat
