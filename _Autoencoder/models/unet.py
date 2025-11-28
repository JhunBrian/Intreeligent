import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------
# UNet Encoder
# ---------------------
class UNetEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        # Downsampling path
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True)
        )  # Output: base_channels

        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(True)
        )  # Output: base_channels*2

        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*2, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(True)
        )  # Output: base_channels*4

        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*4, base_channels*8, 3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(True),
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(True)
        )  # Output: base_channels*8

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels*8, base_channels*16, 3, padding=1),
            nn.BatchNorm2d(base_channels*16),
            nn.ReLU(True),
            nn.Conv2d(base_channels*16, base_channels*16, 3, padding=1),
            nn.BatchNorm2d(base_channels*16),
            nn.ReLU(True)
        )  # Output: base_channels*16

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.bottleneck(x4)
        return x1, x2, x3, x4, x5


# ---------------------
# UNet Decoder
# ---------------------
class UNetDecoder(nn.Module):
    def __init__(self, out_channels=3, base_channels=64):
        super().__init__()

        self.up4 = nn.ConvTranspose2d(base_channels*16, base_channels*8, 2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(base_channels*16, base_channels*8, 3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(True),
            nn.Conv2d(base_channels*8, base_channels*8, 3, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(True)
        )

        self.up3 = nn.ConvTranspose2d(base_channels*8, base_channels*4, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(True),
            nn.Conv2d(base_channels*4, base_channels*4, 3, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(True)
        )

        self.up2 = nn.ConvTranspose2d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(True),
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(True)
        )

        self.up1 = nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True)
        )

        self.final = nn.Conv2d(base_channels, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4, x5):
        d4 = self.up4(x5)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        out = self.sigmoid(out)
        return out


# ---------------------
# UNet Autoencoder
# ---------------------
class UNetAutoencoder(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder(out_channels=out_channels)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        out = self.decoder(x1, x2, x3, x4, x5)
        return out
