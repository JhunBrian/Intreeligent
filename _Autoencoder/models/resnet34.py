import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels=3, pretrained=True, freeze=False):
        super().__init__()

        # Load pretrained weights or empty model
        base = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)

        # Replace first conv if custom input channels
        if in_channels != 3:
            base.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

        # Keep encoder layers only (remove avgpool + fc)
        self.encoder = nn.Sequential(*list(base.children())[:-2])

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.out_channels = 512  # resnet34 final feature depth

    def forward(self, x):
        return self.encoder(x)


class ResNet34Decoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, activation="sigmoid"):
        super().__init__()

        # Define optional final activation
        if activation == "sigmoid":
            self.final_act = nn.Sigmoid()
        elif activation == "tanh":
            self.final_act = nn.Tanh()
        else:
            self.final_act = nn.Identity()

        # Full decoder stack
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.decoder(x)
        return self.final_act(x)


class ResNet34Autoencoder(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        pretrained=True,
        freeze_encoder=False,
        activation="sigmoid"
    ):
        super().__init__()
        self.encoder = ResNet34Encoder(
            in_channels=in_channels,
            pretrained=pretrained,
            freeze=freeze_encoder
        )
        self.decoder = ResNet34Decoder(
            in_channels=self.encoder.out_channels,
            out_channels=out_channels,
            activation=activation
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out
