"""Bandingkan transform evaluasi & TTA multi-skala pada model yang SUDAH ada (tanpa training ulang).

  python eval_variants.py --submission submission --data my_val
  python eval_variants.py --submission submission --data my_val --device dml    # GPU AMD (DirectML), jika terpasang

'squash' = Resize persegi (seperti sekarang). 'crop' = Resize sisi pendek lalu CenterCrop.
'multi-skala' = rata-rata probabilitas dari beberapa resolusi squash (satu model, tanpa bobot tambahan).
my_val hanya 410 gambar: ganti hanya jika unggul jelas (>= ~3 gambar) dari baseline squash 288.
"""
import argparse, importlib.util
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

ap = argparse.ArgumentParser()
ap.add_argument("--submission", default="submission")
ap.add_argument("--data", default="my_val")
ap.add_argument("--device", default="auto", choices=["auto", "cuda", "dml", "cpu"])
ap.add_argument("--dml_index", type=int, default=None)
a = ap.parse_args()


def pick_device(req):
    if req in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda"), "CUDA/ROCm"
    if req in ("auto", "dml"):
        try:
            import torch_directml
            i = a.dml_index
            return torch_directml.device(i), f"DirectML {torch_directml.device_name(i or 0)}"
        except Exception:
            pass
    return torch.device("cpu"), "CPU"


sub = Path(a.submission)
spec = importlib.util.spec_from_file_location("sub_model", sub / "model.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
dev, dev_name = pick_device(a.device)
print(f"device: {dev_name}")
net = mod.get_model()
net.load_state_dict(torch.load(sub / "model.pth", map_location="cpu", weights_only=True), strict=True)
net.to(dev).eval()
norm = transforms.Normalize(mod.MEAN, mod.STD)


def make_tf(kind, resize, crop):
    first = transforms.Resize((resize, resize) if kind == "squash" else resize,
                              interpolation=InterpolationMode.BICUBIC)
    ops = [first] + ([transforms.CenterCrop(crop)] if kind == "crop" else [])
    return transforms.Compose(ops + [transforms.ToTensor(), norm])


def macro_f1(y, p):
    f = []
    for k in range(5):
        tp = ((p == k) & (y == k)).sum(); fp = ((p == k) & (y != k)).sum(); fn = ((p != k) & (y == k)).sum()
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f))


variants = [("squash", 256, None), ("squash", 288, None), ("squash", 320, None),
            ("crop", 288, 288), ("crop", 320, 288), ("crop", 329, 288), ("crop", 352, 288)]
store, Y = {}, None
rows = []


def report(name, probs):
    pred = probs.argmax(1)
    row = (name, macro_f1(Y, pred), int((pred != Y).sum()))
    rows.append(row)
    print(f"{name:34s} Macro-F1 {row[1]:.4f} | salah {row[2]}/{len(Y)}", flush=True)


for kind, rs, cr in variants:
    ds = datasets.ImageFolder(a.data, make_tf(kind, rs, cr))
    assert ds.classes == mod.CLASSES, ds.classes
    PR, YY = [], []
    with torch.no_grad():
        for x, y in DataLoader(ds, 16, shuffle=False, num_workers=0):
            PR.append(net(x.to(dev)).float().softmax(1).cpu()); YY.append(y)
    store[(kind, rs, cr)] = torch.cat(PR).numpy()
    Y = torch.cat(YY).numpy()
    report(f"{kind} resize={rs} crop={cr}", store[(kind, rs, cr)])

sq = lambda s: ("squash", s, None)
for name, keys in {"multi-skala squash 256+288+320": [sq(256), sq(288), sq(320)],
                   "multi-skala squash 288+320": [sq(288), sq(320)]}.items():
    report(name, np.mean([store[k] for k in keys], 0))

base = next(r for r in rows if r[0] == "squash resize=288 crop=None")
best = max(rows, key=lambda r: (r[1], -r[2]))
print(f"\nBaseline (squash 288): Macro-F1 {base[1]:.4f}, salah {base[2]}. Terbaik: {best[0]} "
      f"(Macro-F1 {best[1]:.4f}, salah {best[2]}).")
if base[2] - best[2] >= 3:
    print("Unggul jelas (>= 3 gambar). Kirim hasil ini ke saya; perubahan get_transform()/model.py perlu ditulis "
          "sesuai pemenangnya, lalu divalidasi ulang dengan evaluate.py.")
else:
    print("Tidak ada keunggulan yang jelas -> biarkan get_transform() seperti sekarang.")