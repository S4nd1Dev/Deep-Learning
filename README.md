# Studio 2 — Klasifikasi Sampah · Kelompok Mahamihi

Deep Learning (IF25-40401) · Teknik Informatika ITERA · Semester Ganjil 2026/2027

## Anggota Kelompok

| No | Nama Lengkap | NIM | Kontribusi utama |
|----|--------------|-----|------------------|
| 1  | Muhammad Ghama Al Fajri | 123140182 | Membangun model awal dan update transform untuk meningkatkan performa |
| 2  | Abi Sholihan | 123140192 | Menyesuaikan parameter dan training untuk meningkatkan performa |
| 3  | Riyan Sandi Prayoga | 123140176 | Diskusi opsi peningkatan dan validasi independen |

## Ringkasan Pendekatan

- **Arsitektur / backbone**: ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`, timm), pretrained ImageNet-22k→1k, fine-tuning penuh dengan head 5 kelas. Satu model tunggal (tanpa ensemble).
- **Strategi pelatihan**: AdamW (LR backbone 1e-4, head 1e-3, weight decay 0,05), warmup 1 epoch + cosine, 15 epoch, batch 16, EMA bobot (0,998), drop-path 0,1, label smoothing 0,1, MixUp+CutMix (peluang 0,5; dimatikan 3 epoch terakhir), mixed precision. Augmentasi: RandomResizedCrop (skala 0,35–1), flip horizontal, TrivialAugmentWide, RandomErasing. Checkpoint terbaik (bobot mentah atau EMA, per epoch) dipilih dari Macro-F1 pada split lokal 15% (`my_val`, 410 gambar, seed 42, identik dengan split `train_baseline.py`).
- **Data tambahan**: tidak ada (hanya train set resmi).
- **Ukuran input & preprocessing**: training 288×288. Evaluasi: Resize sisi pendek ke 288 (bicubic, rasio asli terjaga) lalu CenterCrop 288, normalisasi ImageNet. Test-time augmentation flip horizontal ada di dalam `forward()`. Bobot disimpan fp16 (~56 MB), sehingga tidak butuh Git LFS.
- **Perangkat pelatihan**: GPU CUDA dengan VRAM 6 GB dan RAM 16 GB (batch 16, workers 0), sekitar 40 detik per epoch.

## Hasil

| Putaran | Commit | Macro-F1 (split validasi lokal) | Macro-F1 (diumumkan asisten) |
|---------|--------|----------------------------------|------------------------------|
| Preview 1 (14 Okt) | – | – | tidak dinilai (belum ada submission) |
| Preview 2 (21 Okt) | – | 0,9657 | 0,9649  |
| Final (25 Okt)     | – | – | – |

Skor lokal diukur dengan `evaluate.py` pada `my_val` (410 gambar). Pada ukuran sekecil itu, satu gambar salah mengubah akurasi sekitar 0,25 poin, dan hasil antar-run dengan resep yang sama bisa berbeda sekitar 0,01, jadi selisih kecil tidak kami anggap sebagai perbaikan nyata.

### Catatan eksperimen
- **Run 1** (CPU; training 256, batch 32, MixUp 0,8): Macro-F1 lokal 0,9657; model ini yang dinilai di Preview 2 (0,9649).
- **Run 2** (GPU; training 288, batch 16, MixUp 0,5): Macro-F1 lokal 0,9735 dengan transform evaluasi squash 288×288. Model inilah yang bobotnya dipakai sekarang. Tiga perubahan (resolusi, batch, MixUp) dilakukan bersamaan, sehingga kami tidak bisa memastikan mana yang berperan.
- **Transform evaluasi** (bobot sama, tanpa training ulang): mengganti squash 288×288 dengan Resize sisi pendek + CenterCrop menaikkan Macro-F1 lokal dari 0,9735 menjadi 0,9813 (salah 10 → 7 dari 410). Keempat varian crop yang diuji (288, 320→288, 329→288, 352→288) lebih baik daripada semua varian squash (256, 288, 320). Rata-rata prediksi multi-skala (256+288+320) tidak membantu (0,9703), jadi tidak dipakai. Skrip pengujiannya: `eval_variants.py`.
- **Sweep resolusi evaluasi** (squash, Run 2): 256 → 0,9636; 288 → 0,9735; 320 → 0,9712. Karena itu `EVAL_SIZE = 288`.
- **Run 3** (retrain dengan resep Run 2, validasi memakai transform crop): `evaluate.py` 0,9784 (salah 8 dari 410). Tidak lebih baik daripada model Run 2 (selisih satu gambar), sehingga `model.pth` tidak diganti.
- **Variansi antar-run**: resep yang sama menghasilkan 0,9712 (GPU) dan 0,9608 (CPU) pada epoch terakhir, jadi satu run saja tidak cukup untuk menilai perubahan kecil.
- **Referensi**: ide menguji Resize+CenterCrop terinspirasi dari pengamatan terhadap repo publik tim lain (mereka melaporkan squash lebih buruk). Implementasi dan pengujiannya kami lakukan sendiri pada split lokal kami. Tidak ada kode, bobot, atau prediksi kelompok lain yang dipakai.

## Cara Reproduksi

```bash
pip install -r requirements.txt
python train.py --data ..\train        # membuat my_train/ dan my_val/ lalu menyimpan checkpoint terbaik ke submission/model.pth
python evaluate.py --submission submission --data my_val
python eval_variants.py --submission submission --data my_val   # opsional: bandingkan transform evaluasi
```

Dataset tidak disertakan di repo ini (dilarang diunggah ulang). `model.pth` yang ada di repo dihasilkan oleh Run 2, sebelum `train.py` menyesuaikan cara membaca ukuran evaluasi untuk transform crop. Proses pelatihannya identik; yang berubah hanya cara skor validasi per epoch dihitung.

## Struktur Repositori

```
submission/
  model.py          # get_model() dan get_transform() (Resize + CenterCrop)
  model.pth         # state_dict model terbaik (fp16)
  requirements.txt  # timm
evaluate.py         # salinan skrip evaluasi resmi (tidak diubah)
train.py            # kode pelatihan
eval_variants.py    # uji varian transform evaluasi
requirements.txt
```