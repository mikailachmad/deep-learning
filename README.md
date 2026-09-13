# Single Layer Perceptron | Klasifikasi Biner Iris (Setosa vs Versicolor)

Implementasi Python dari Single Layer Perceptron (SLP) untuk tugas mata kuliah #1 Pembelajaran Mesin Mendalam. Kode ini memakai langkah perhitungan yang ada di spreadsheet (forward pass, sigmoid, backpropagation, update bobot per sampel), termasuk skema pembagian data training & validasi per epoch, sehingga hasil akurasi & loss-nya sama dengan hasil yang ada di Sheet.

## Struktur Repo

```
slp-iris-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── iris.csv              # 100 baris: 50 Iris-setosa + 50 Iris-versicolor
├── src/
│   ├── data_loader.py        # load & susun urutan data training/validasi
│   ├── model.py               # SLP: forward pass, sigmoid, backprop per sampel
│   └── train.py               # loop training 5 epoch + validasi, simpan hasil
└── outputs/
    ├── results.csv            # rekap loss & akurasi per epoch (train & val)
    ├── loss_chart.png         # grafik loss (dipakai di PPT halaman 5)
    └── accuracy_chart.png     # grafik akurasi (dipakai di PPT halaman 3)
```

Kenapa dipecah begini:

- `data_loader.py` dan `model.py` dipisah agar logika data pada epoch tidak bercampur dengan logika matematika SLP sehingga mudah diperiksa saat ada yang beda dari spreadsheet.

## Cara menjalankan

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

## Metodologi

- **Bobot awal:** bias = teta1 = teta2 = teta3 = teta4 = 0.5
- **Learning rate:** 0.1
- **Aktivasi:** sigmoid, `g(z) = 1 / (1 + e^-z)`
- **Update bobot:** per sampel (stochastic gradient descent), bukan per batch
- **Training** (bobot berjalan menerus, tidak reset antar epoch):
  - Epoch 1: 80 sampel (40 Setosa + 40 Versicolor)
  - Epoch 2-5: 100 sampel/epoch, melanjutkan siklus 80 sampel core yang sama secara terus-menerus (tanpa reset di batas epoch)
- **Validasi** (bobot dicabang dari bobot training di akhir epoch itu, lalu ikut terupdate sepanjang epoch validasi, tidak memengaruhi bobot training):
  - Epoch 1 & 2: 20 sampel yang disisihkan dari training (10 Setosa + 10 Versicolor yang tidak dipakai training)
  - Epoch 3-5: seluruh 100 sampel dataset

## Hasil

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| ----- | ---------- | --------- | -------- | ------- |
| 1     | 0.4499     | 52.5%     | 0.1001   | 85.0%   |
| 2     | 0.0386     | 95.0%     | 0.0355   | 95.0%   |
| 3     | 0.0176     | 98.0%     | 0.0110   | 99.0%   |
| 4     | 0.0171     | 98.0%     | 0.0085   | 99.0%   |
| 5     | 0.0081     | 100.0%    | 0.0072   | 100.0%  |

## Sumber data

`data/iris.csv` diekspor dari sheet `Data` pada spreadsheet SLP (Iris dataset, subset Setosa & Versicolor saja untuk klasifikasi biner).

## Penulis

Nama: Mikail Achmad  
NIM: 24/542370/PA/23026  
Kelas: KOM - B  
Tugas: Pembelajaran Mesin Mendalam | Assignment 1 (Single Layer Perceptron)
