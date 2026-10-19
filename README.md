# Studio 2 — Klasifikasi Sampah · Kelompok Mahamihi

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Muhammad Ghama Al Fajri | 123140182 | Membangun model awal |
| 2  | Abi Sholihan | 123140192 | ISI |
| 3  | Riyan Sandi Prayoga | 123140176 | ISI |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`, timm), pretrained ImageNet-22k→1k, fine-tuning penuh dengan head 5 kelas.
- **Strategi pelatihan**: AdamW (LR backbone 1e-4, head 1e-3, weight decay 0,05), warmup 1 epoch + cosine, 15 epoch, EMA bobot (0,998), drop-path 0,1, label smoothing 0,1, MixUp+CutMix (dimatikan 3 epoch terakhir), augmentasi RandomResizedCrop + TrivialAugment + RandomErasing, mixed precision. Model dipilih dari Macro-F1 pada split lokal 15% (sama dengan baseline).
- **Data tambahan**: tidak ada. _(Ubah jika memakai data luar, cantumkan sumber + jumlahnya.)_
- **Ukuran input & preprocessing**: training 256×256, evaluasi 288×288 (resize bicubic), normalisasi ImageNet, test-time augmentation flip horizontal di dalam `forward()`. Bobot disimpan fp16 (~56 MB) sehingga tidak butuh Git LFS.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | – | – | – |
| Preview 2 (21 Okt) | | | |
| Final (25 Okt)     | | | |

Catatan eksperimen singkat (apa yang dicoba, apa yang berhasil/gagal):
- 

## Cara Reproduksi

```bash
pip install -r requirements.txt
python train.py --data ..\train     # membuat my_train/ my_val/ dan submission/model.pth
python evaluate.py --submission submission --data ..\my_val
```

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform()
  model.pth         # state_dict model terbaik (fp16)
  requirements.txt  # timm
evaluate.py         # salinan skrip evaluasi resmi (tidak diubah)
train.py            # kode pelatihan
requirements.txt
```
