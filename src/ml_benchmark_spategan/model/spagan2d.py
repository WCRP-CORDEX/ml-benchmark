from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import amp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDropout(nn.Module):
    def __init__(self, p: float, d_seed: int):
        super().__init__()
        self.p = p
        torch.manual_seed(d_seed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch, channels, height, width = x.shape

        mask_shape = (batch, channels, height, width)
        mask = torch.bernoulli(torch.ones(mask_shape, device=device) * (1 - self.p))
        mask = mask.repeat(1, 1, 1, 1) / (1 - self.p)

        return x * mask


class ResidualBlock2D(nn.Module):
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
            nn.ReflectionPad2d((1, 1, 1, 1)) if padding_type else None
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.norm1 = nn.InstanceNorm2d(out_channels, affine=False)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=1,
            padding=padding,
            bias=False,
        )
        self.norm2 = nn.InstanceNorm2d(out_channels, affine=False)

        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels or stride != 1:
            self.adjust_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.adjust_norm = nn.InstanceNorm2d(out_channels)
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
    def __init__(self, scale_factor: tuple, mode: str = "bilinear"):
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
    def __init__(self, cf):
        super().__init__()

        self.filter_size = cf.filter_size
        self.n_input_channels = cf.n_input_channels
        self.n_output_channels = cf.n_output_channels
        self.dropout_seed = cf.dropout_seed
        self.dropout_ratio = cf.dropout_ratio
        self._initialize_layers()

    def _initialize_layers(self):
        f = self.filter_size

        self.res1 = ResidualBlock2D(self.n_input_channels, f, use_layer_norm=False, padding_type=True)
        self.res2 = ResidualBlock2D(f, f, use_layer_norm=False, padding_type=True)
        self.res3 = ResidualBlock2D(f, f, use_layer_norm=True, padding_type=True)

        self.down0 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(f, f, kernel_size=(3, 3), stride=(2,2), padding=0),
            nn.ReLU(inplace=True),
        )

        self.upu1 = Interpolate((2, 2))
        self.res3b = ResidualBlock2D(f, f, padding_type=True)

        self.up0 = Interpolate((2, 2))
        self.res4 = ResidualBlock2D(f, f, padding_type=True)
        self.up1 = Interpolate((2, 2))
        self.res5 = ResidualBlock2D(f, f, padding_type=True)

        self.up2 = Interpolate((1, 1))
        self.res6 = ResidualBlock2D(f, f, padding_type=True)

        self.up3 = Interpolate((3, 3))
        self.res7 = ResidualBlock2D(f, f, padding_type=True)

        self.up4 = Interpolate((2, 2))
        self.res8 = ResidualBlock2D(f, f, padding_type=True)
        self.res9 = ResidualBlock2D(f, f, use_layer_norm=False, padding_type=True)

        self.output_conv = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(f, self.n_output_channels, kernel_size=(3, 3), padding=0),
        )
        # self.linout = nn.Linear(128*128, 128*128)

        self.constraint_layer = Constraint()

    def forward(self, x: torch.Tensor, dropout_seed: int = None) -> torch.Tensor:
        if dropout_seed is None:
            dropout_seed = self.dropout_seed
        x1 = self.res1(x)
        x1 = CustomDropout(p=self.dropout_ratio, d_seed=dropout_seed)(x1)
        x2_stay = self.res2(x1)
        # x2_stay = x1

        x2 = self.down0(x2_stay)
        x2 = self.res3b(x2)
        x2 = CustomDropout(p=self.dropout_ratio, d_seed=dropout_seed)(x2)
        x2 = self.upu1(x2)

        x2 = x2_stay + x2
        x2 = self.res3(x2)
        x2 = CustomDropout(p=self.dropout_ratio, d_seed=dropout_seed)(x2)

        x2 = self.up0(x2)
        x2 = self.res4(x2)

        x2 = self.up1(x2)
        x2 = self.res5(x2)
        x2 = CustomDropout(p=self.dropout_ratio, d_seed=dropout_seed)(x2)

        x2 = self.up4(x2)
        x2 = self.res8(x2)
        x2 = self.res9(x2)

        output = self.output_conv(x2)
        output = torch.flatten(output, start_dim=1)
        # output = self.linout(output)


        # Avoid in-place operation to prevent gradient computation error
        # output = output[:, 0, :, :]
        # output_constrained[:, :, :, 32:-32, 32:-32] = self.constraint_layer(
        #     output[:, :, :, 32:-32, 32:-32], x
        # )

        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.apply(self._init_weights)
        
        self.int_reflection = nn.ReflectionPad3d((1, 1, 1, 1, 0, 0))

        # HIGH RESOLUTION layers:
        self.conv1 = ResidualBlock2D(1, 128, use_layer_norm=False, stride=(1,1), )
        self.conv2 = ResidualBlock2D(128, 128, use_layer_norm=True, stride=(2,2),)
        self.conv3 = ResidualBlock2D(128, 128, use_layer_norm=True, stride=(2,2), )
        self.conv4 = ResidualBlock2D(128, 64, use_layer_norm=True, stride=(3,3), )
        self.conv5 = ResidualBlock2D(128, 64, use_layer_norm=True, stride=(2,2), )
        
        
        # LOW RESOLUTION layers
        self.conv1_1 = ResidualBlock2D(2, 64, use_layer_norm=False, stride=(1,1),)
        self.conv1_2 = ResidualBlock2D(64, 32, use_layer_norm=False, stride=(2,2),)
        
        self.conv_combined = ResidualBlock2D(96, 64, use_layer_norm=True, stride=(2,2),)

        # Output convolution
        self.output_conv = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.trunc_normal_(m.weight, std=.02)
            # nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    

    def forward(self, x, y):
        
        y = y[:,[0,2]]
        #y = self.int_reflection(y)
        
        noisex = torch.randn(x.size()).cuda() * 0.05
        noisey = torch.randn(y.size()).cuda() * 0.05
        
        
        x = x + noisex
        y = y + noisey
        
        
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        x11 = self.conv1_1(y)
        x12 = self.conv1_2(x11)
        
        xy = torch.cat((x4, x12), dim=1)
        
        xy = self.conv_combined(xy)
        
        out = self.output_conv(xy)
        
        return out

def train_gan_step(config, input_image, target, step, discriminator, generator, gen_opt, disc_opt, scaler, criterion):
    generator.train()
    discriminator.train()

    gen_opt.zero_grad(set_to_none=True)
    
    # mixed precission
    with amp.autocast():

        ##################
        ### Generator: ###
        ##################
        
        ## generate multiple ensemble prediction-
        gen_ensemble = torch.cat([generator(input_image) for _ in range(3)], dim=1)
        pred_log = gen_ensemble[:,0:1]

        # calculate ensemble mean
        gen_ensemble = (gen_ensemble[:,0:1] + gen_ensemble[:,1:2]  + gen_ensemble[:,2:3]) / 3
                

         # Classify all fake batch with D
        disc_fake_output = discriminator(pred_log[:, 0:1], input_image)
        
        gen_gan_loss = criterion(disc_fake_output, torch.ones_like(disc_fake_output))
        l1loss = nn.L1Loss()(gen_ensemble, target)
        loss = (l1loss  + gen_gan_loss)

    scaler.scale(loss).backward()
    # Gradient Norm Clipping
    #nn.utils.clip_grad_norm_(generator.parameters(), max_norm=2.0, norm_type=2)
    
    scaler.step(gen_opt)
    scaler.update()
    # Unscale gradients to prevent underflow - 
    #scaler.unscale_(gen_opt)
 
    ####################
    ## Discriminator: ##
    ####################
    disc_opt.zero_grad(set_to_none=True)
    
    pred_log = pred_log[:, 0:1].detach()
    
    with amp.autocast():

        # discriminator prediction
        disc_real_output = discriminator(target, input_image)
        disc_real = criterion(disc_real_output, torch.ones_like(disc_real_output))


        # Classify all fake batch with D
        disc_fake_output = discriminator(pred_log, input_image) 
        
        # Calculate D's loss on the all-fake batch
        disc_fake = criterion(disc_fake_output, torch.zeros_like(disc_fake_output))

    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
    scaler.scale(disc_fake + disc_real).backward()
    
    # Gradient Norm Clipping
    #nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=2.0, norm_type=2)
    scaler.step(disc_opt)
    scaler.update()
    # Unscale gradients to prevent underflow - dont know if really necessary
    #scaler.unscale_(gen_opt)
    



if __name__ == "__main__":
    from torchinfo import summary

    model = Generator().to(device)
    summary(model, input_size=(1, 15, 16, 16))
