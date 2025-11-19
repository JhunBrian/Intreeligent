import torch
import torch.nn as nn
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models import resnet50, resnet101, resnet152
from torchvision.ops import FeaturePyramidNetwork

class BottleneckFPNBackbone(nn.Module):
    def __init__(self, variant='resnet50', pretrained=True, out_channels=256):
        """
        Args:
            variant (str): 'resnet50', 'resnet101', 'resnet152'
            pretrained (bool): Load pretrained ImageNet weights
            out_channels (int): Number of channels in FPN output
        """
        super().__init__()
        assert variant in ['resnet50', 'resnet101', 'resnet152'], "Only ResNet-50, 101, 152 supported"

        # Load backbone
        if variant == 'resnet50':
            resnet = resnet50(pretrained=pretrained)
        elif variant == 'resnet101':
            resnet = resnet101(pretrained=pretrained)
        elif variant == 'resnet152':
            resnet = resnet152(pretrained=pretrained)

        # Channels at each stage of the ResNet with bottleneck blocks
        self.in_channels_list = [256, 512, 1024, 2048]

        # Bottom-up layers
        self.body = nn.ModuleDict({
            'conv1': resnet.conv1,
            'bn1': resnet.bn1,
            'relu': resnet.relu,
            'maxpool': resnet.maxpool,
            'layer1': resnet.layer1,
            'layer2': resnet.layer2,
            'layer3': resnet.layer3,
            'layer4': resnet.layer4,
        })

        # FPN
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.in_channels_list,
            out_channels=out_channels
        )

        # Required by MaskRCNN
        self.out_channels = out_channels

    def forward(self, x):
        x = self.body['conv1'](x)
        x = self.body['bn1'](x)
        x = self.body['relu'](x)
        x = self.body['maxpool'](x)

        c1 = self.body['layer1'](x)
        c2 = self.body['layer2'](c1)
        c3 = self.body['layer3'](c2)
        c4 = self.body['layer4'](c3)

        features = self.fpn({
            '0': c1,
            '1': c2,
            '2': c3,
            '3': c4
        })
        return features


class TreeMaskRCNN(nn.Module):
    def __init__(self, backbone_variant='resnet50', num_classes=2, pretrained=True):
        """
        Args:
            backbone_variant (str): 'resnet50', 'resnet101', 'resnet152'
            num_classes (int): number of classes including background
            pretrained (bool): load pretrained weights for backbone
        """
        super().__init__()

        backbone = BottleneckFPNBackbone(variant=backbone_variant, pretrained=pretrained)

        # RPN anchor generator
        anchor_generator = AnchorGenerator(
            sizes=((32,), (64,), (128,), (256,)),
            aspect_ratios=((0.5, 1.0, 2.0),) * 4
        )

        self.model = MaskRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)
