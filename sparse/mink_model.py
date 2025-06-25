import MinkowskiEngine as ME
import torch.nn as nn


class SparseConvBlock(nn.Module):
    """Two 3×3 sub-manifold sparse convolutions (+BN+ReLU)."""

    def __init__(self, in_ch: int, out_ch: int, D: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            ME.MinkowskiConvolution(in_ch, out_ch, kernel_size=3, dimension=D),
            ME.MinkowskiBatchNorm(out_ch),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(out_ch, out_ch, kernel_size=3, dimension=D),
            ME.MinkowskiBatchNorm(out_ch),
            ME.MinkowskiReLU(inplace=True),
        )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        return self.block(x)


class SparseUNet(nn.Module):
    """
    Sparse 3D U-Net equivalent to the dense UNet you provided.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 2, D: int = 3):
        super().__init__()
        # Encoder
        self.enc1 = SparseConvBlock(in_channels, 64, D)
        self.down1 = ME.MinkowskiConvolution(
            64, 64, kernel_size=2, stride=2, bias=False, dimension=D
        )  # replaces MaxPool3d

        self.enc2 = SparseConvBlock(64, 128, D)
        self.down2 = ME.MinkowskiConvolution(
            128, 128, kernel_size=2, stride=2, bias=False, dimension=D
        )

        self.enc3 = SparseConvBlock(128, 256, D)
        self.down3 = ME.MinkowskiConvolution(
            256, 256, kernel_size=2, stride=2, bias=False, dimension=D
        )

        self.enc4 = SparseConvBlock(256, 512, D)
        self.down4 = ME.MinkowskiConvolution(
            512, 512, kernel_size=2, stride=2, bias=False, dimension=D
        )

        # Bottleneck
        self.bottleneck = SparseConvBlock(512, 1024, D)

        # Decoder
        self.up4 = ME.MinkowskiConvolutionTranspose(
            1024, 512, kernel_size=2, stride=2, bias=False, dimension=D
        )
        self.dec4 = SparseConvBlock(1024, 512, D)

        self.up3 = ME.MinkowskiConvolutionTranspose(
            512, 256, kernel_size=2, stride=2, bias=False, dimension=D
        )
        self.dec3 = SparseConvBlock(512, 256, D)

        self.up2 = ME.MinkowskiConvolutionTranspose(
            256, 128, kernel_size=2, stride=2, bias=False, dimension=D
        )
        self.dec2 = SparseConvBlock(256, 128, D)

        self.up1 = ME.MinkowskiConvolutionTranspose(
            128, 64, kernel_size=2, stride=2, bias=False, dimension=D
        )
        self.dec1 = SparseConvBlock(128, 64, D)

        # Final 1×1 sparse conv
        self.final = ME.MinkowskiConvolution(
            64, out_channels, kernel_size=1, bias=True, dimension=D
        )

    def forward(self, x: ME.SparseTensor) -> ME.SparseTensor:
        # Encoder path
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        b = self.bottleneck(self.down4(e4))

        # Decoder path with skip connections (use ME.cat)
        d4 = self.dec4(ME.cat(self.up4(b), e4))
        d3 = self.dec3(ME.cat(self.up3(d4), e3))
        d2 = self.dec2(ME.cat(self.up2(d3), e2))
        d1 = self.dec1(ME.cat(self.up1(d2), e1))

        return self.final(d1)
