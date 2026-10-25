"""Submission kelompok Mahamihi: ConvNeXt-Tiny (pretrained ImageNet-22k -> 1k) + TTA flip.

Sesuai kontrak evaluate.py:
  - get_model() hanya MEMBANGUN arsitektur (tanpa unduh bobot, tanpa memuat model.pth).
  - get_transform() -> PIL.Image (RGB) -> Tensor [3, H, W].
  - Output: logits [batch, 5]; urutan kelas alfabetis (sama dengan ImageFolder):
      0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam
evaluate.py memuat model.pth dengan strict=True & weights_only=True, jadi nama parameter
harus persis sama dengan yang disimpan train.py (keduanya memakai kelas Net ini).
"""
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ARCH = "convnext_tiny.fb_in22k_ft_in1k"
NUM_CLASSES = 5
CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EVAL_SIZE = 288  # sesuaikan dengan hasil sweep resolusi di train.py (256 / 288 / 320)


class Net(nn.Module):
    def __init__(self, pretrained=False, drop_path=0.0):
        super().__init__()
        # pretrained=True hanya dipakai oleh train.py. get_model() selalu pretrained=False.
        self.backbone = timm.create_model(
            ARCH, pretrained=pretrained, num_classes=NUM_CLASSES, drop_path_rate=drop_path
        )

    def forward(self, x):
        out = self.backbone(x)
        if not self.training:  # TTA: rata-rata logits citra asli + flip horizontal
            out = (out + self.backbone(torch.flip(x, dims=[3]))) / 2
        return out


def get_model():
    return Net(pretrained=False)


def get_transform(size=None):
    size = size or EVAL_SIZE
    return transforms.Compose([
        transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])