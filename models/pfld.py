import torch
from torch import nn, Tensor
import math

from typing import Any, Callable, List, Optional


class Conv2dNormActivation(nn.Sequential):
    """Convolutional block, consists of nn.Conv2d, nn.BatchNorm2d, nn.ReLU"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional = None,
            groups: int = 1,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: int = 1,
            inplace: Optional[bool] = True,
            bias: bool = False,
    ) -> None:

        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation

        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(num_features=out_channels, eps=0.001, momentum=0.01)
        ]

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        super().__init__(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, expand_ratio: int) -> None:
        super().__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        hidden_dim = int(round(in_planes * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_planes == out_planes

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                Conv2dNormActivation(
                    in_planes,
                    hidden_dim,
                    kernel_size=1,
                    activation_layer=nn.ReLU
                )
            )
        layers.extend(
            [
                # dw
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    activation_layer=nn.ReLU,
                ),
                # pw-linear
                nn.Conv2d(hidden_dim, out_planes, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_planes),
            ]
        )
        self.conv = nn.Sequential(*layers)
        self.out_channels = out_planes
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDInference(nn.Module):
    def __init__(self, num_classes: int = 196) -> None:
        super().__init__()

        self.stage1 = nn.Sequential(
            Conv2dNormActivation(in_channels=3, out_channels=64, kernel_size=3, stride=2),
            Conv2dNormActivation(in_channels=64, out_channels=64, kernel_size=3, stride=1),  # [64, 56, 56]
        )

        self.stage2 = nn.Sequential(
            InvertedResidual(in_planes=64, out_planes=64, stride=2, expand_ratio=2),
            InvertedResidual(in_planes=64, out_planes=64, stride=1, expand_ratio=2),
            InvertedResidual(in_planes=64, out_planes=64, stride=1, expand_ratio=2),
            InvertedResidual(in_planes=64, out_planes=64, stride=1, expand_ratio=2),
            InvertedResidual(in_planes=64, out_planes=64, stride=1, expand_ratio=2),  # [64, 28, 28]
        )

        self.stage3 = nn.Sequential(
            InvertedResidual(in_planes=64, out_planes=128, stride=2, expand_ratio=2),
            InvertedResidual(in_planes=128, out_planes=128, stride=1, expand_ratio=4),
            InvertedResidual(in_planes=128, out_planes=128, stride=1, expand_ratio=4),
            InvertedResidual(in_planes=128, out_planes=128, stride=1, expand_ratio=4),
            InvertedResidual(in_planes=128, out_planes=128, stride=1, expand_ratio=4),
            InvertedResidual(in_planes=128, out_planes=128, stride=1, expand_ratio=4),
            InvertedResidual(in_planes=128, out_planes=128, stride=1, expand_ratio=4),
            InvertedResidual(in_planes=128, out_planes=16, stride=1, expand_ratio=2),  # [16, 14, 14]
        )

        self.stage4 = Conv2dNormActivation(in_channels=16, out_channels=32, kernel_size=3, stride=2)  # [32, 7, 7]

        self.stage5 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=7, stride=1, padding=0),  # [128, 1, 1]
            nn.ReLU(inplace=True),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(176, num_classes)

    def forward(self, x):
        x = self.stage1(x)  # [b, 64, 56, 56]
        x = self.stage2(x)  # [b, 64, 28, 28]
        out = x
        x = self.stage3(x)  # [b, 16, 14, 14]

        x1 = self.avg_pool(x)  # [b, 16, 1, 1]
        x1 = torch.flatten(x1, 1)

        x = self.stage4(x)  # [b, 32, 7, 7]

        x2 = self.avg_pool(x)  # [b, 32, 1, 1]
        x2 = torch.flatten(x2, 1)

        x3 = self.stage5(x)  # [b, 128, 1, 1]
        x3 = torch.flatten(x3, 1)

        multi_scale_features = torch.cat([x1, x2, x3], 1)

        landmarks = self.fc(multi_scale_features)

        return out, landmarks


class AuxiliaryNet(nn.Module):
    """
    AuxiliaryNet is a lightweight CNN designed for 3-class classification.
    """

    def __init__(self):
        super().__init__()

        # Convolutional layers for feature extraction
        self.conv1 = Conv2dNormActivation(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.conv2 = Conv2dNormActivation(in_channels=128, out_channels=128, kernel_size=3, stride=1)
        self.conv3 = Conv2dNormActivation(in_channels=128, out_channels=32, kernel_size=3, stride=2)
        self.conv4 = Conv2dNormActivation(in_channels=32, out_channels=128, kernel_size=7, stride=1)

        # Adaptive average pooling to ensure fixed-size output
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor

        x = self.fc1(x)
        x = self.fc2(x)

        return x


if __name__ == '__main__':
    input = torch.randn(1, 3, 112, 112)
    pfld_backbone = PFLDInference()

    print("num params", sum(p.numel() for p in pfld_backbone.parameters() if p.requires_grad))

    auxiliarynet = AuxiliaryNet()
    features, landmarks = pfld_backbone(input)
    angle = auxiliarynet(features)

    print("angle.shape:{0:}, landmarks.shape: {1:}".format(
        angle.shape, landmarks.shape))
