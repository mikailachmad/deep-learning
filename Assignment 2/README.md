# Klasifikasi MNIST dengan MLP (Keras) | Regularization & Early Stopping

Implementasi Python (Jupyter Notebook) untuk tugas mata kuliah Pembelajaran Mesin Mendalam. Notebook ini mereplikasi kode MLP untuk klasifikasi digit MNIST dari materi _Learning Strategy: Neural Network_ (slide Strategi Pembelajaran, halaman 14–23), lalu dilanjutkan dengan eksperimen penerapan L1/L2 regularization, Dropout, dan Early Stopping di atas arsitektur yang sama.

## Struktur Repo

```
Assignment 2/
├── README.md
├── main.ipynb                      # notebook bersih (belum dieksekusi)
├── main_after_assignment.ipynb     # notebook yang sudah dijalankan (berisi semua output & grafik)
└── output/
    └── my_model1.keras             # model MLP baseline (Bagian 2) hasil model1.save()
```

## Cara menjalankan

Notebook dijalankan di Google Colab :

1. Buka `main.ipynb` di Google Colab.
2. Jalankan seluruh cell dari atas ke bawah (`Runtime > Run all`). Dataset MNIST otomatis diunduh lewat `keras.datasets.mnist.load_data()`.
3. Semua model, grafik, dan tabel perbandingan akan tergenerate ulang di setiap run (bobot inisialisasi acak, jadi angka bisa sedikit berbeda tiap kali dijalankan, tapi tren antar eksperimen konsisten).

Untuk melihat hasil tanpa menjalankan ulang, buka langsung `main_after_assignment.ipynb`.

## Struktur Notebook

### Bagian 1 | Contoh implementasi ANN (Iris, 3 kelas)

Contoh `Sequential` + `Dense(softmax)` dari slide, dijalankan di atas dataset Iris (3 kelas) agar benar-benar bisa dieksekusi (arsitektur: `Dense(64, relu) → Dense(3, softmax)`).

### Bagian 2 | Replikasi MLP MNIST dari slide halaman 14–23

Arsitektur `Flatten(28,28,1) → Dense(64, relu) → Dense(10, softmax)`, dilatih 10 epoch, `batch_size=100`, optimizer `adam`, loss `categorical_crossentropy`. Model disimpan ke `output/my_model1.keras` lalu dimuat ulang untuk prediksi.

### Bagian 3 | Eksperimen Regularization & Early Stopping

Arsitektur dasar yang sama dipakai di semua eksperimen agar perbandingan adil. Model (kecuali Early Stopping) dilatih **20 epoch** untuk melihat gap overfitting train-val yang menunjukkan efek tiap teknik regularisasi.

| #   | Eksperimen        | Rate/Setting                                                              | Alasan pemilihan                                                                                               |
| --- | ----------------- | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| a   | L1 Regularization | `kernel_regularizer=l1(1e-4)` pada `Dense(64)`                            | Titik awal wajar untuk bobot dense; rate lebih besar (mis. 1e-2) bisa membuat banyak bobot collapse ke 0       |
| b   | L2 Regularization | `kernel_regularizer=l2(1e-4)` pada `Dense(64)`                            | Disamakan magnitude-nya dengan L1 agar perbandingan efek "shrink ke 0" (L1) vs "shrink proporsional" (L2) adil |
| c   | Dropout           | `Dropout(0.3)` setelah `Dense(64)`                                        | Nilai moderat (umum 0.2–0.5) untuk MLP kecil berunit 64                                                        |
| d   | Early Stopping    | `monitor='val_loss', patience=3, restore_best_weights=True`, max 30 epoch | Epoch maksimum dinaikkan ke 30 agar callback benar-benar punya ruang untuk trigger                             |
| e   | L1 + Dropout      | Gabungan (a) + (c)                                                        | Rate sama seperti eksperimen individunya, untuk melihat efek kombinasi                                         |

Setiap eksperimen menampilkan `model.summary()`, akurasi/loss test, serta loss chart & accuracy chart (train vs val). Bagian 3.7 merangkum semuanya dalam satu tabel + grafik perbandingan gabungan.

## Hasil

**Bagian 1 (Iris, 3 kelas):** Test accuracy 100%, test loss 0.1018.

**Bagian 2 (MLP MNIST, replikasi slide, 10 epoch):** Test accuracy 97.18%, test loss 0.0870.

**Bagian 3 (eksperimen regularisasi & early stopping, 20 epoch):**

| Model          | Test Acc | Test Loss | Gap Val-Train Loss (epoch akhir) |
| -------------- | -------- | --------- | -------------------------------- |
| Baseline       | 97.58%   | 0.0910    | 0.0666                           |
| L1             | 97.49%   | 0.1589    | 0.0295                           |
| L2             | 97.63%   | 0.1070    | 0.0487                           |
| Dropout        | 97.32%   | 0.0941    | -0.0096                          |
| Early Stopping | 97.46%   | 0.0846    | 0.0773                           |
| L1 + Dropout   | 97.08%   | 0.1852    | -0.0458                          |

## Kesimpulan

- **Dropout** dan **L1+Dropout** paling efektif menekan overfitting. Gap val-train loss keduanya negatif (val loss sedikit di bawah train loss), menandakan model tidak menghafal data training secara berlebihan.
- Penekanan overfitting ini **tidak mengorbankan akurasi secara berarti**, seluruh model tetap di rentang 97.1%–97.6% test accuracy, selisih Baseline ke L1+Dropout (kombinasi paling ketat) sebesar ~0.5%.
- **Early Stopping** berhenti di epoch 20 dari maksimum 30 dan **test loss terendah** (0.0846) di antara semua model.
- **L1** dan **L2** murni sama-sama mengecilkan gap dibanding Baseline, tapi lebih lemah dibanding Dropout dalam menekan overfitting pada eksperimen ini.

## Penulis

Nama: Mikail Achmad  
NIM: 24/542370/PA/23026  
Kelas: KOM - B  
Tugas: Pembelajaran Mesin Mendalam — Assignment 2 (MLP MNIST + Regularization & Early Stopping)
