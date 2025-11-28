import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    resnet34, resnet50, resnet101, resnet152,
    ResNet34_Weights, ResNet50_Weights,
    ResNet101_Weights, ResNet152_Weights
)


# ---------------------------------------------------------
# AutoPad: Ensures H,W divisible by 32 and removes padding later
# ---------------------------------------------------------

class AutoPad2d(nn.Module):
    """
    Pads input so H and W are divisible by 32 for ResNet encoders.
    Stores padding values and removes them after decoding.
    """
    def __init__(self):
        super().__init__()
        self.pad = None

    def encode_pad(self, x):
        h, w = x.shape[-2:]
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32

        self.pad = (0, pad_w, 0, pad_h)   # (left, right, top, bottom)
        return F.pad(x, self.pad)

    def decode_unpad(self, x):
        left, right, top, bottom = self.pad
        H, W = x.shape[-2:]

        return x[..., top:H-bottom if bottom > 0 else H,
                 left:W-right if right > 0 else W]


# ---------------------------------------------------------
# ResNet Configuration Table
# ---------------------------------------------------------

RESNET_CONFIGS = {
    "resnet34":  (resnet34,  ResNet34_Weights.DEFAULT,  512),
    "resnet50":  (resnet50,  ResNet50_Weights.DEFAULT, 2048),
    "resnet101": (resnet101, ResNet101_Weights.DEFAULT, 2048),
    "resnet152": (resnet152, ResNet152_Weights.DEFAULT, 2048),
}


# ---------------------------------------------------------
# ResNet Encoder (supports 34, 50, 101, 152)
# ---------------------------------------------------------

class ResNetEncoder(nn.Module):
    def __init__(self, model_name="resnet34", in_channels=3, pretrained=True, freeze=False):
        super().__init__()
        assert model_name in RESNET_CONFIGS, f"Model must be one of {list(RESNET_CONFIGS.keys())}"

        constructor, weights, out_ch = RESNET_CONFIGS[model_name]

        base = constructor(weights=weights if pretrained else None)

        # Replace first conv if custom input channels
        if in_channels != 3:
            base.conv1 = nn.Conv2d(
                in_channels, 64,
                kernel_size=7, stride=2, padding=3, bias=False
            )

        # Remove avgpool and fc
        self.encoder = nn.Sequential(*list(base.children())[:-2])
        self.out_channels = out_ch

        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def forward(self, x):
        return self.encoder(x)


# ---------------------------------------------------------
# Decoder (generic for all ResNet backbones)
# ---------------------------------------------------------

class GenericDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, activation="sigmoid"):
        super().__init__()

        if activation == "sigmoid":
            self.final_act = nn.Sigmoid()
        elif activation == "tanh":
            self.final_act = nn.Tanh()
        else:
            self.final_act = nn.Identity()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 512, 2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, out_channels, 2, 2),
        )

    def forward(self, x):
        return self.final_act(self.decoder(x))


# ---------------------------------------------------------
# Full Autoencoder (Encoder + Decoder + AutoPad)
# ---------------------------------------------------------

class ResNetAutoencoder(nn.Module):
    def __init__(
        self,
        model_name="resnet34",
        in_channels=3,
        out_channels=3,
        pretrained=True,
        freeze_encoder=False,
        activation="sigmoid",
    ):
        super().__init__()

        self.autopad = AutoPad2d()

        self.encoder = ResNetEncoder(
            model_name=model_name,
            in_channels=in_channels,
            pretrained=pretrained,
            freeze=freeze_encoder
        )

        self.decoder = GenericDecoder(
            in_channels=self.encoder.out_channels,
            out_channels=out_channels,
            activation=activation,
        )

    def forward(self, x):
        # 1. Pad input to make divisible by 32
        x_pad = self.autopad.encode_pad(x)

        # 2. Encode
        z = self.encoder(x_pad)

        # 3. Decode
        out = self.decoder(z)

        # 4. Remove padding → original size
        out = self.autopad.decode_unpad(out)

        return out