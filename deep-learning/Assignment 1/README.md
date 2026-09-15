# Single Layer Perceptron - Klasifikasi Biner Iris (Setosa vs Versicolor)

Implementasi Python dari **Single Layer Perceptron (SLP)** untuk tugas mata kuliah Pembelajaran Mesin Mendalam. Kode ini mereplikasi persis logika perhitungan yang ada di spreadsheet (forward pass, sigmoid, backpropagation, update bobot per sampel), sehingga hasil akurasi & loss-nya **sama dengan hasil di spreadsheet**.

## Struktur Repo

```
slp-iris-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── PMM-TemplateSLP_update_fixed.xlsx   # sumber data (sheet "Data")
├── src/
│   └── train.py                 #  Kode untuk load data, model SLP, training loop, output
└── outputs/
    ├── results.csv              # rekap loss & akurasi per epoch (train & val)
    ├── loss_chart.png           # grafik loss gabungan (train + val, 1 grafik)
    └── accuracy_chart.png       # grafik akurasi gabungan (train + val, 1 grafik)
```

## Cara menjalankan

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

Data dibaca langsung dari file Excel (`data/PMM-TemplateSLP_update_fixed.xlsx`, sheet `Data`) pakai `pandas.read_excel`.

## Metodologi

- **Bobot awal:** bias = teta1 = teta2 = teta3 = teta4 = 0.5
- **Learning rate:** 0.1
- **Aktivasi:** sigmoid, `g(z) = 1 / (1 + e^-z)`
- **Update bobot:** per sampel (stochastic gradient descent), bukan per batch

### Pembagian data (split memakai pandas)

Dataset (`sheet Data`, 100 baris) difilter berdasarkan label, lalu dislice:

| Set        | Isi                                                                                       | Jumlah         |
| ---------- | ----------------------------------------------------------------------------------------- | -------------- |
| Training   | 40 Setosa pertama + 40 Versicolor pertama                                                 | 80 baris/epoch |
| Validation | 10 Setosa berikutnya + 10 Versicolor berikutnya (held-out, tidak pernah dilihat training) | 20 baris/epoch |

Set yang sama dipakai berulang di setiap 5 epoch.

- **Training**: bobot berjalan menerus (tidak reset antar epoch), satu garis lurus gradient descent sepanjang 5×80 = 400 update.
- **Validation**: bobot dicabang dari bobot training di akhir epoch itu, ikut terupdate sepanjang 20 sampel validasi epoch tsb, tapi tidak memengaruhi bobot training, setiap epoch validasi selalu mulai dari cabang baru.

## Hasil

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| ----- | ---------- | --------- | -------- | ------- |
| 1     | 0.4499     | 52.5%     | 0.1001   | 85.0%   |
| 2     | 0.0375     | 95.0%     | 0.0651   | 90.0%   |
| 3     | 0.0244     | 97.5%     | 0.0458   | 95.0%   |
| 4     | 0.0174     | 97.5%     | 0.0346   | 100.0%  |
| 5     | 0.0127     | 98.75%    | 0.0278   | 100.0%  |

## Sumber data

`data/PMM-TemplateSLP_update_fixed.xlsx` : spreadsheet SLP awal (Iris dataset, subset Setosa & Versicolor untuk klasifikasi biner).

## Penulis

Nama: Mikail Achmad  
NIM: 24/542370/PA/23026  
Kelas: KOM - B  
Tugas: Pembelajaran Mesin Mendalam — Assignment 1 (Single Layer Perceptron)
