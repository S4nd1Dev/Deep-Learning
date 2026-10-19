"""Training Studio 2 (Mahamihi): fine-tuning ConvNeXt-Tiny, tahan overfitting & label noise.

Dijalankan dari folder repo (sejajar dengan evaluate.py), sama seperti train_baseline.py:

  python train.py --data ../train --epochs 15
  python evaluate.py --submission ./submission --data ./my_val        # uji resmi

Split lokal 15% (my_train/ & my_val/, seed 42) IDENTIK dengan train_baseline.py, jadi skor
bisa dibandingkan langsung dengan baseline ResNet-18. Jika my_val/ sudah ada, dipakai ulang.

Sebelum Final (opsional): latih ulang di SEMUA data dengan epoch yang sama
  python train.py --data ../train --epochs 15 --full
"""
import argparse, contextlib, csv, math, os, random, shutil, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEmaV2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submission.model import CLASSES, MEAN, NUM_CLASSES, STD, Net, get_model, get_transform


def make_local_split(src, dst, frac, seed):  # sama persis dengan train_baseline.py
    dst_tr, dst_va = Path(dst) / "my_train", Path(dst) / "my_val"
    if dst_va.exists():
        return dst_tr, dst_va
    rng = random.Random(seed)
    for cls_dir in sorted(Path(src).iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.glob("*.jpg"))
        rng.shuffle(files)
        n_va = int(len(files) * frac)
        for i, f in enumerate(files):
            out = (dst_va if i < n_va else dst_tr) / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / f.name)
    return dst_tr, dst_va


def macro_f1(y, p, n=NUM_CLASSES):
    f1s = []
    for k in range(n):
        tp = np.sum((p == k) & (y == k)); fp = np.sum((p == k) & (y != k)); fn = np.sum((p != k) & (y == k))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return float(np.mean(f1s)), f1s


def train_tf(size):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.35, 1.0), ratio=(0.75, 1.333),
                                     interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, value="random"),
    ])


def save_fp16(model, path):
    # fp16 -> ~56 MB (< batas 100 MB GitHub): tidak perlu Git LFS / Release.
    # evaluate.py: load_state_dict(strict=True) otomatis meng-cast ke fp32.
    sd = {k: (v.detach().half().cpu() if v.is_floating_point() else v.detach().cpu())
          for k, v in model.state_dict().items()}
    torch.save(sd, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../train", help="folder train/ dari dataset")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--size", type=int, default=256, help="resolusi training")
    ap.add_argument("--lr", type=float, default=1e-4, help="LR backbone")
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop_path", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--nomix_last", type=int, default=3, help="epoch terakhir tanpa mixup/cutmix")
    ap.add_argument("--full", action="store_true", help="latih di my_train + my_val (tanpa seleksi holdout)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    a = ap.parse_args()
    out = a.out or ("submission/model_full.pth" if a.full else "submission/model.pth")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    cuda = dev.type == "cuda"
    amp_dtype = torch.bfloat16 if (cuda and torch.cuda.is_bf16_supported()) else torch.float16
    amp = (lambda: torch.autocast("cuda", dtype=amp_dtype)) if cuda else contextlib.nullcontext
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and amp_dtype == torch.float16)

    tr_dir, va_dir = make_local_split(a.data, ".", 0.15, a.seed)
    tr_ds = datasets.ImageFolder(tr_dir, train_tf(a.size))
    assert tr_ds.classes == CLASSES, tr_ds.classes
    if a.full:
        train_ds = ConcatDataset([tr_ds, datasets.ImageFolder(va_dir, train_tf(a.size))])
    else:
        train_ds = tr_ds
    print(f"device={dev} | train={len(train_ds)} | holdout(my_val)={len(list(Path(va_dir).rglob('*.jpg')))}")

    train_dl = DataLoader(train_ds, a.bs, shuffle=True, num_workers=a.workers, pin_memory=cuda,
                          drop_last=True, persistent_workers=a.workers > 0)

    def val_loader(size):
        ds = datasets.ImageFolder(va_dir, get_transform(size))
        return ds, DataLoader(ds, 64, shuffle=False, num_workers=a.workers, pin_memory=cuda)

    @torch.no_grad()
    def predict(m, loader):
        m.eval(); P, Y = [], []
        for x, y in loader:
            with amp():
                o = m(x.to(dev, non_blocking=True))
            P.append(o.float().softmax(1).cpu()); Y.append(y)
        return torch.cat(P).numpy(), torch.cat(Y).numpy()

    model = Net(pretrained=True, drop_path=a.drop_path).to(dev)
    ema = ModelEmaV2(model, decay=a.ema)
    head = [p for n, p in model.named_parameters() if n.startswith("backbone.head")]
    body = [p for n, p in model.named_parameters() if not n.startswith("backbone.head")]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr, "base": a.lr},
                             {"params": head, "lr": a.head_lr, "base": a.head_lr}], weight_decay=a.wd)
    mixup = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0, prob=0.8, switch_prob=0.5,
                  label_smoothing=0.1, num_classes=NUM_CLASSES)
    soft_ce, ls_ce = SoftTargetCrossEntropy(), LabelSmoothingCrossEntropy(0.1)

    spe = len(train_dl); total = a.epochs * spe; warm = spe; step = 0
    best, best_ep = -1.0, 0
    _, val_dl = val_loader(get_transform().transforms[0].size[0])

    for ep in range(a.epochs):
        model.train(); t0 = time.time(); run = 0.0
        use_mix = ep < a.epochs - a.nomix_last
        for x, y in train_dl:
            f = (step + 1) / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = g["base"] * max(f, 0.01)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            if use_mix:
                x, t = mixup(x, y); crit = soft_ce
            else:
                t, crit = y, ls_ce
            with amp():
                loss = crit(model(x), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); ema.update(model)
            run += loss.item(); step += 1
        msg = f"ep {ep+1:02d}/{a.epochs} loss {run/spe:.4f} ({time.time()-t0:.0f}s)"
        if not a.full:
            res = {}
            for name, m in (("raw", model), ("ema", ema.module)):
                pr, yy = predict(m, val_dl)
                res[name] = (macro_f1(yy, pr.argmax(1))[0], m)
            msg += " | " + " ".join(f"{k} F1 {v[0]:.4f}" for k, v in res.items())
            name = max(res, key=lambda k: res[k][0])
            if res[name][0] > best:
                best, best_ep = res[name][0], ep + 1
                save_fp16(res[name][1], out); msg += f"  <- best ({name}), disimpan"
        print(msg, flush=True)

    if a.full:
        save_fp16(ema.module, out)
        print(f"[full] EMA epoch terakhir -> {out}. Skor my_val TIDAK valid (my_val ikut training). "
              f"Untuk dipakai: ganti nama menjadi submission/model.pth.")
        return

    print(f"\nBest holdout Macro-F1 {best:.4f} @ epoch {best_ep} -> {out}")
    chk = get_model().to(dev)  # verifikasi persis seperti evaluate.py
    chk.load_state_dict(torch.load(out, map_location=dev, weights_only=True), strict=True)
    print("Sweep resolusi evaluasi (set EVAL_SIZE di submission/model.py ke yang terbaik):")
    for s in (256, 288, 320):
        _, dl = val_loader(s)
        pr, yy = predict(chk, dl)
        f, per = macro_f1(yy, pr.argmax(1))
        print(f"  size {s}: Macro-F1 {f:.4f} | acc {(pr.argmax(1)==yy).mean():.4f} | F1/kelas {[round(v,3) for v in per]}")
    ds, dl = val_loader(get_transform().transforms[0].size[0])
    pr, yy = predict(chk, dl)
    with open("holdout_errors.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["path", "label", "pred", "conf"])
        for j, (p, _) in enumerate(ds.samples):
            if pr[j].argmax() != yy[j]:
                w.writerow([p, CLASSES[yy[j]], CLASSES[pr[j].argmax()], f"{pr[j].max():.3f}"])
    print("salah-prediksi holdout -> holdout_errors.csv (cek label yang meragukan)")


if __name__ == "__main__":
    main()
