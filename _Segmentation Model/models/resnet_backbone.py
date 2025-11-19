import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152
)
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.ops.misc import Conv2dNormActivation


# -------------------------------------------------------
# Utility: Build a ResNet (returns the full model)
# -------------------------------------------------------
def _build_resnet(name, pretrained):
    name = name.lower()
    if name in ["r18", "resnet18"]:
        return resnet18(weights="IMAGENET1K_V1" if pretrained else None)
    elif name in ["r34", "resnet34"]:
        return resnet34(weights="IMAGENET1K_V1" if pretrained else None)
    elif name in ["r50", "resnet50"]:
        return resnet50(weights="IMAGENET1K_V1" if pretrained else None)
    elif name in ["r101", "resnet101"]:
        return resnet101(weights="IMAGENET1K_V1" if pretrained else None)
    elif name in ["r152", "resnet152"]:
        return resnet152(weights="IMAGENET1K_V1" if pretrained else None)
    else:
        raise ValueError(f"Unknown backbone name: {name}")


# -------------------------------------------------------
# C4: Use only the conv4 (res4) output
# -------------------------------------------------------
class ResNetC4(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        # Use layers until res4
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool, resnet.layer1, resnet.layer2
        )
        self.res4 = resnet.layer3  # C4 output

    def forward(self, x):
        x = self.stem(x)
        x = self.res4(x)
        return {"res4": x}


# -------------------------------------------------------
# DC5: Replace layer4 stride with dilation
# -------------------------------------------------------
class ResNetDC5(nn.Module):
    def __init__(self, resnet):
        super().__init__()
        # Make DC5 modifications
        for n, m in resnet.layer4.named_modules():
            if "conv2" in n:
                m.dilation = (2, 2)
                m.padding = (2, 2)
                m.stride = (1, 1)

        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool, resnet.layer1, resnet.layer2
        )
        self.res4 = resnet.layer3
        self.res5 = resnet.layer4  # DC5 modified

    def forward(self, x):
        x = self.stem(x)
        c4 = self.res4(x)
        c5 = self.res5(c4)
        return {"res4": c4, "res5": c5}


# -------------------------------------------------------
# FPN: Build a Feature Pyramid on C2, C3, C4, C5
# -------------------------------------------------------
class ResNetFPN(nn.Module):
    def __init__(self, resnet, out_channels=256):
        super().__init__()
        # Bottom-up
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # C2
        self.layer2 = resnet.layer2  # C3
        self.layer3 = resnet.layer3  # C4
        self.layer4 = resnet.layer4  # C5
        
        # Channels for each stage
        if isinstance(resnet, (resnet50().__class__, resnet101().__class__, resnet152().__class__)):
            channels = [256, 512, 1024, 2048]  # Bottleneck
        else:
            channels = [64, 128, 256, 512]     # Basic block

        # FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=channels,
            out_channels=out_channels
        )

    def forward(self, x):
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return self.fpn({"0": c2, "1": c3, "2": c4, "3": c5})
        # Outputs: { "0":P2, "1":P3, "2":P4, "3":P5 }


# -------------------------------------------------------
# Main Wrapper Class
# -------------------------------------------------------
class ResNetBackbone(nn.Module):
    """
    Example:
        backbone = ResNetBackbone("R50", variant="FPN", pretrained=True)
        feats = backbone(images)
    """
    def __init__(self, name="R50", variant="FPN", pretrained=True, out_channels=256):
        super().__init__()
        base = _build_resnet(name, pretrained=pretrained)

        variant = variant.upper()
        if variant == "C4":
            self.backbone = ResNetC4(base)
        elif variant == "DC5":
            self.backbone = ResNetDC5(base)
        elif variant == "FPN":
            self.backbone = ResNetFPN(base, out_channels=out_channels)
        else:
            raise ValueError(f"Unknown variant: {variant}")

    def forward(self, x):
        return self.backbone(x)
