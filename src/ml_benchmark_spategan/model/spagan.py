from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDropout(nn.Module):
    def __init__(self, p: float, d_seed: int):
        super().__init__()
        self.p = p
        torch.manual_seed(d_seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch, channels, time, height, width = x.shape

        mask_shape = (batch, channels, 1, height, width)
        mask = torch.bernoulli(torch.ones(mask_shape, device=device) * (1 - self.p))
        mask = mask.repeat(1, 1, time, 1, 1) / (1 - self.p)

        return x * mask


class ResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_layer_norm: bool = True,
        stride: int = 1,
        padding_type: Optional[bool] = None,
    ):
        super().__init__()

        padding = 0 if padding_type else 1
        self.use_layer_norm = use_layer_norm
        self.padding_type = padding_type

        self.padding_layer = (
            nn.ReflectionPad3d((1, 1, 1, 1, 0, 0)) if padding_type else None
        )

        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=False)

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=1,
            padding=padding,
            bias=False,
        )
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=False)

        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.adjust_conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.adjust_norm = nn.InstanceNorm3d(out_channels)
        else:
            self.adjust_conv = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if self.padding_layer:
            x = self.padding_layer(x)

        out = self.conv1(x)
        if self.use_layer_norm:
            out = self.norm1(out)
        out = self.relu(out)

        if self.padding_layer:
            out = self.padding_layer(out)

        out = self.conv2(out)
        if self.use_layer_norm:
            out = self.norm2(out)

        if self.adjust_conv:
            residual = self.adjust_conv(residual)
            residual = self.adjust_norm(residual)

        out += residual
        return self.relu(out)


class Interpolate(nn.Module):
    def __init__(self, scale_factor: tuple, mode: str = "trilinear"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False
        )


class Constraint(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, prediction, constraint):
        constraint = constraint[:, :, 5:-5, 8:-8, 8:-8].sum(dim=1, keepdim=True)
        scale = (constraint[:, 0].mean(dim=(1, 2, 3)) / 6).view(-1, 1, 1, 1, 1)
        pred_mean = prediction[:, 0].mean(dim=(1, 2, 3)).view(-1, 1, 1, 1, 1)
        return prediction * (scale / pred_mean)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.filter_size = 32
        self.n_input_channels = 15
        self._initialize_layers()

    def _initialize_layers(self):
        f = self.filter_size

        self.res1 = ResidualBlock3D(self.n_input_channels, f, use_layer_norm=False, padding_type=True)
        # self.res2 = ResidualBlock3D(f, f, use_layer_norm=False, padding_type=True)
        self.res3 = ResidualBlock3D(f, f, use_layer_norm=True, padding_type=True)

        self.down0 = nn.Sequential(
            nn.ReflectionPad3d((1, 1, 1, 1, 0, 0)),
            nn.Conv3d(f, f, kernel_size=(1, 3, 3), stride=(1,2,2), padding=0),
            nn.ReLU(inplace=True),
        )

        self.upu1 = Interpolate((1, 2, 2))
        self.res3b = ResidualBlock3D(f, f, padding_type=True)

        self.up0 = Interpolate((1, 2, 2))
        self.res4 = ResidualBlock3D(f, f, padding_type=True)

        self.up1 = Interpolate((1, 2, 2))
        self.res5 = ResidualBlock3D(f, f, padding_type=True)

        self.up2 = Interpolate((1, 1, 1))
        self.res6 = ResidualBlock3D(f, f, padding_type=True)

        self.up3 = Interpolate((1, 3, 3))
        self.res7 = ResidualBlock3D(f, f, padding_type=True)

        self.up4 = Interpolate((1, 2, 2))
        self.res8 = ResidualBlock3D(f, f, padding_type=True)
        self.res9 = ResidualBlock3D(f, f, use_layer_norm=False, padding_type=True)

        self.output_conv = nn.Sequential(
            nn.ReflectionPad3d((1, 1, 1, 1, 0, 0)),
            nn.Conv3d(f, 1, kernel_size=(1, 3, 3), padding=0),
            # nn.Linear(),
        )

        self.constraint_layer = Constraint()

    def forward(self, x: torch.Tensor, dropout_seed: int = 1234) -> torch.Tensor:
        x1 = self.res1(x)
        # x1 = CustomDropout(p=0.2, d_seed=dropout_seed)(x1)
        # x2_stay = self.res2(x1)
        x2_stay = x1

        x2 = self.down0(x2_stay)
        x2 = self.res3b(x2)
        # x2 = CustomDropout(p=0.2, d_seed=dropout_seed)(x2)
        x2 = self.upu1(x2)

        x2 = x2_stay + x2
        x2 = self.res3(x2)
        # x2 = CustomDropout(p=0.2, d_seed=dropout_seed)(x2)

        x2 = self.up0(x2)
        x2 = self.res4(x2)

        x2 = self.up1(x2)
        x2 = self.res5(x2)
        # x2 = CustomDropout(p=0.2, d_seed=dropout_seed)(x2)

        x2 = self.up2(x2)
        x2 = self.res6(x2)

        # x2 = self.up3(x2)
        # x2 = self.res7(x2)

        x2 = self.up4(x2)
        x2 = self.res8(x2)
        x2 = self.res9(x2)

        output = self.output_conv(x2)

        # Avoid in-place operation to prevent gradient computation error
        output = output[:, 0, 0, :, :]
        # output_constrained[:, :, :, 32:-32, 32:-32] = self.constraint_layer(
        #     output[:, :, :, 32:-32, 32:-32], x
        # )

        return output


if __name__ == "__main__":
    from torchinfo import summary

    model = Generator().to(device)
    summary(model, input_size=(1, 15, 1, 16, 16))
