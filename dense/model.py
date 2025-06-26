import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=2):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool3d(2)
        self.bottleneck = self.conv_block(512, 1024)

        self.up4 = self.up_conv(1024, 512)
        self.dec4 = self.conv_block(1024, 512)
        self.up3 = self.up_conv(512, 256)
        self.dec3 = self.conv_block(512, 256)
        self.up2 = self.up_conv(256, 128)
        self.dec2 = self.conv_block(256, 128)
        self.up1 = self.up_conv(128, 64)
        self.dec1 = self.conv_block(128, 64)

        self.final = nn.Conv3d(64, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def up_conv(self, in_ch, out_ch):
        return nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)

    def center_crop(self, enc_feat, target_spatial):
        _, _, d, h, w = enc_feat.shape
        td, th, tw = target_spatial
        start_d = (d - td) // 2
        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        return enc_feat[:, :, start_d:start_d+td, start_h:start_h+th, start_w:start_w+tw]

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.final(d1)

# def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool(e1))
#         e3 = self.enc3(self.pool(e2))
#         e4 = self.enc4(self.pool(e3))
#         b = self.bottleneck(self.pool(e4))

#         u4 = self.up4(b)
#         e4 = self.center_crop(e4, u4.shape[2:])
#         d4 = self.dec4(torch.cat([u4, e4], dim=1))

#         u3 = self.up3(d4)
#         e3 = self.center_crop(e3, u3.shape[2:])
#         d3 = self.dec3(torch.cat([u3, e3], dim=1))

#         u2 = self.up2(d3)
#         e2 = self.center_crop(e2, u2.shape[2:])
#         d2 = self.dec2(torch.cat([u2, e2], dim=1))

#         u1 = self.up1(d2)
#         e1 = self.center_crop(e1, u1.shape[2:])
#         d1 = self.dec1(torch.cat([u1, e1], dim=1))

#         return self.final(d1)