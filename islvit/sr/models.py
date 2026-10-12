"""Networks for the super-resolution GAN front end.

Every clip reaches the hand detector through ``crops.read_sampled_frames``, which
brings any video below the 720p working height up to it with bicubic
interpolation. That line is where the project's low-resolution failures pass:
CISLR's 300 px frames are upscaled 2.4x before MediaPipe looks for a hand, and a
phone or smart-glasses camera would be too. The generator here replaces that one
interpolation; the discriminator only exists during training.

The design follows Real-ESRGAN (Wang et al., ICCVW 2021) at a size this project
can afford:

* **Generator: a compact VGG-style network** (Real-ESRGAN's SRVGGNetCompact). All
  convolutions run at the *low* resolution and a final pixel shuffle produces the
  x4 output, so full frames are cheap; the network predicts a residual over a
  nearest-neighbour upsample, so an untrained or unsure network degrades towards
  plain upscaling rather than towards noise.
* **Discriminator: a U-Net with spectral normalisation.** It returns a realness
  score per pixel rather than per image, which is what blind super-resolution
  needs: a crop can be convincing in the palm and wrong at the fingertips, and a
  single scalar cannot say where.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class SRGenerator(nn.Module):
    """x``scale`` super-resolution; input and output are RGB in [0, 1]."""

    def __init__(self, channels: int = 3, features: int = 64, convs: int = 16, scale: int = 4) -> None:
        super().__init__()
        self.scale = scale
        self.config = {"channels": channels, "features": features, "convs": convs, "scale": scale}
        layers: list[nn.Module] = [nn.Conv2d(channels, features, 3, 1, 1), nn.PReLU(features)]
        for _ in range(convs):
            layers += [nn.Conv2d(features, features, 3, 1, 1), nn.PReLU(features)]
        layers.append(nn.Conv2d(features, channels * scale * scale, 3, 1, 1))
        self.body = nn.Sequential(*layers)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return base + self.shuffle(self.body(x))


class UNetDiscriminatorSN(nn.Module):
    """Per-pixel real/fake logits, U-Net shaped, spectrally normalised."""

    def __init__(self, channels: int = 3, features: int = 64) -> None:
        super().__init__()
        f = features
        sn = spectral_norm
        self.conv0 = nn.Conv2d(channels, f, 3, 1, 1)
        self.conv1 = sn(nn.Conv2d(f, f * 2, 4, 2, 1, bias=False))
        self.conv2 = sn(nn.Conv2d(f * 2, f * 4, 4, 2, 1, bias=False))
        self.conv3 = sn(nn.Conv2d(f * 4, f * 8, 4, 2, 1, bias=False))
        self.conv4 = sn(nn.Conv2d(f * 8, f * 4, 3, 1, 1, bias=False))
        self.conv5 = sn(nn.Conv2d(f * 4, f * 2, 3, 1, 1, bias=False))
        self.conv6 = sn(nn.Conv2d(f * 2, f, 3, 1, 1, bias=False))
        self.conv7 = sn(nn.Conv2d(f, f, 3, 1, 1, bias=False))
        self.conv8 = sn(nn.Conv2d(f, f, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(f, 1, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        act = lambda t: F.leaky_relu(t, 0.2, inplace=True)
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)
        x0 = act(self.conv0(x))
        x1 = act(self.conv1(x0))
        x2 = act(self.conv2(x1))
        x3 = act(self.conv3(x2))
        x4 = act(self.conv4(up(x3))) + x2
        x5 = act(self.conv5(up(x4))) + x1
        x6 = act(self.conv6(up(x5))) + x0
        out = act(self.conv7(x6))
        out = act(self.conv8(out))
        return self.conv9(out)


class VGGPerceptualLoss(nn.Module):
    """L1 distance between VGG19 features of output and target (Real-ESRGAN weights).

    Features are taken *before* each ReLU, as in BasicSR, so negative responses are
    compared too. ImageNet weights come from torchvision; if they cannot be
    fetched, ResNet-50 (already in the local torch cache) stands in, and the
    choice is recorded in the run summary.
    """

    LAYERS = {2: 0.1, 7: 0.1, 16: 1.0, 25: 1.0, 34: 1.0}  # conv1_2 conv2_2 conv3_4 conv4_4 conv5_4

    def __init__(self) -> None:
        super().__init__()
        import torchvision

        try:
            vgg = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1)
            self.backbone = "vgg19"
            self.features = vgg.features[: max(self.LAYERS) + 1].eval()
        except Exception as error:  # offline: fall back to a cached backbone
            print(f"  VGG19 weights unavailable ({error}); using ResNet-50 features")
            resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
            self.backbone = "resnet50"
            self.features = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                          resnet.layer1, resnet.layer2, resnet.layer3).eval()
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _taps(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = (x - self.mean) / self.std
        if self.backbone != "vgg19":
            taps = []
            for index, layer in enumerate(self.features):
                x = layer(x)
                if index >= 4:
                    taps.append(x)
            return taps
        taps = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.LAYERS:
                taps.append(x)
        return taps

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weights = list(self.LAYERS.values()) if self.backbone == "vgg19" else [1.0, 1.0, 1.0]
        loss = output.new_zeros(())
        for weight, a, b in zip(weights, self._taps(output), self._taps(target)):
            loss = loss + weight * F.l1_loss(a, b)
        return loss


def count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
