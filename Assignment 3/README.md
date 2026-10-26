# RNN Lab: Next-Word Prediction & Poem Generation (Shakespeare's Sonnets)

Implementasi Python (Jupyter Notebook) untuk tugas mata kuliah Pembelajaran Mesin Mendalam. Notebook ini mengimplementasikan sel **RNN secara manual** (tanpa `nn.RNN` bawaan PyTorch) untuk next-word prediction pada korpus _Shakespeare's Sonnets_ (154 soneta), lalu memakai model terlatih untuk men-generate puisi baru kata demi kata. Konfigurasi eksperimen (panjang window, ukuran hidden state, learning rate, pembagian data train/test, prompt generate, dan random seed) diturunkan secara deterministik dari NIM mahasiswa.

## Struktur Repo

```
Assignment 3/
├── README.md
├── rnn_lab_starter.ipynb              # notebook final (semua TODO 1-10 + Part C1-C3 terisi, sudah dijalankan)
└── Poem Generation_Mikail Achmad.pdf  # laporan tugas
```

## Cara menjalankan

Notebook dijalankan di Google Colab:

1. Buka `rnn_lab_starter.ipynb` di Google Colab.
2. Jalankan seluruh cell dari atas ke bawah (`Runtime > Run all`). Dataset soneta otomatis diunduh dari repo [martin-gorner/tensorflow-rnn-shakespeare](https://raw.githubusercontent.com/martin-gorner/tensorflow-rnn-shakespeare/master/shakespeare/sonnets.txt).
3. `cfg` (parameter eksperimen) diturunkan otomatis dari `NIM = 23026` lewat `random.Random(nim)`, sehingga setiap mahasiswa mendapat kombinasi angka yang berbeda. Bobot RNN diinisialisasi acak dengan seed dari NIM, jadi angka hasil training akan reproduce sama persis selama cell dijalankan berurutan dari atas ke bawah.

Laporan lengkap (deskripsi tugas, metodologi, hasil, dan analisis Part C) ada di `Poem Generation_Mikail Achmad.pdf`.

## Struktur Notebook

### Konfigurasi (dari NIM 23026)

| Parameter    | Nilai                   | Keterangan                                                                |
| ------------ | ----------------------- | ------------------------------------------------------------------------- |
| `seq_len`    | 20                      | Panjang window konteks (jumlah kata) yang dibaca RNN per langkah training |
| `hidden`     | 128                     | Ukuran hidden state RNN                                                   |
| `emb`        | 64                      | Ukuran embedding kata (fixed)                                             |
| `lr`         | 0.003                   | Learning rate optimizer Adam                                              |
| `batch`      | 64                      | Ukuran mini-batch (fixed)                                                 |
| `epochs`     | 8                       | Jumlah epoch training (fixed)                                             |
| `test_start` | 32                      | 8 soneta berurutan mulai indeks 32 dijadikan test set                     |
| `prompt`     | `['o', 'that', 'time']` | Kata awal untuk proses generate puisi                                     |
| `seed`       | 3026                    | Seed untuk inisialisasi bobot, shuffling, dan sampling                    |

### Step 1-3 | Tokenisasi, Vocabulary, Encoding & Context Window

Teks diubah menjadi huruf kecil, dipecah menjadi token kata (huruf + apostrof, `"love's"` tetap satu token) dan token tanda baca (`, . ; : ! ?`) secara terpisah; karakter lain dibuang. Vocabulary dibangun hanya dari soneta training (kata muncul ≥2 kali masuk vocabulary, sisanya `<unk>`) → **V = 1330** kata unik. Setiap token di-encode jadi id, lalu dibentuk window `(X, Y)` dengan panjang `L = 20` (Y = X digeser satu posisi) → **19.288 window training**, **1.036 window test**.

### Step 4 | RNN Manual (`MyRNN`)

Recurrence `h_t = tanh(e_t · Wxh + h_(t-1) · Whh + bh)` diimplementasikan manual dengan `Wxh`, `Whh`, `bh` sebagai `nn.Parameter`, tanpa `nn.RNN` bawaan PyTorch. Divalidasi menghasilkan output identik (toleransi `1e-5`) dengan `nn.RNN` resmi PyTorch saat diberi bobot yang sama.

### Step 5 | Training, Evaluasi & Baseline

`run_epoch` melakukan shuffle + mini-batch + backpropagation + gradient clipping (norm 1.0) saat training, dan forward pass tanpa gradien saat evaluasi, mengembalikan rata-rata cross-entropy per kata. Sebagai pembanding dihitung **unigram baseline** dengan add-one smoothing: `p(w) = (count_train(w) + 1) / (N_train + V)`.

### Step 6 | Generate Puisi

Model men-generate kata baru satu per satu: logit kata terakhir dibagi _temperature_, logit `<unk>` dipaksa `-1e9` (tidak pernah dipilih), satu kata disampling dengan `torch.multinomial` dan diumpankan balik bersama hidden state. Token `<eos>` dicetak sebagai baris baru.

### Part C | Eksperimen (prediksi jurnal vs hasil aktual)

| Eksperimen                                          | Prediksi            | Hasil Aktual      |
| --------------------------------------------------- | ------------------- | ----------------- |
| **C1** - kata unik dari 50 kata (T=0.5 / 1.0 / 1.5) | 30 / 35 / 45        | 29 / 37 / 48      |
| **C2** - epoch dengan test perplexity terendah      | Epoch 3             | Epoch 2           |
| **C3** - jumlah parameter (RNN manual / `nn.LSTM`)  | ~280.000 / ~350.000 | 281.394 / 356.018 |

## Hasil

**Sanity check sebelum training (CHECK 5):** untrained test loss `7.252` vs `ln(V) = 7.193` (selisih `0.059`, jauh di bawah toleransi `0.7`).

**Training RNN manual (8 epoch):**

| Metode                               | Test Perplexity Terbaik | Epoch Terbaik | Test Perplexity Epoch Terakhir |
| ------------------------------------ | ----------------------- | ------------- | ------------------------------ |
| Unigram baseline (add-one smoothing) | 176.7                   | -             | -                              |
| RNN manual                           | 119.5                   | 2             | 125.4                          |

**Perbandingan RNN manual vs `nn.LSTM` (C3):**

| Model            | Jumlah Parameter | Epoch Terbaik | Test Perplexity Terbaik |
| ---------------- | ---------------- | ------------- | ----------------------- |
| RNN manual       | 281.394          | 2             | 119.5                   |
| `nn.LSTM`        | 356.018          | 2             | 112.9                   |
| Unigram baseline | -                | -             | 176.7                   |

**Contoh hasil generate** (prompt `['o', 'that', 'time']`, temperature=1.0):

> "o that time things my then, some verse youth shall in that respect doth stand, and it might still you no how time lovest; your love is me, but as if new could be my proud in mistress ride. but, home see me for you. ah so is i mine forsake best is they"

## Kesimpulan

- RNN manual (_hand-written_, tanpa `nn.RNN`) berhasil diimplementasikan dan divalidasi identik dengan `nn.RNN` resmi PyTorch, lalu dilatih sampai mengalahkan unigram baseline (119.5 vs 176.7 test perplexity) yang membuktikan model memanfaatkan konteks kata, bukan sekadar frekuensi kata.
- Korpus yang kecil (146 soneta training) membuat model cepat overfitting. Performa terbaik di test set justru tercapai pada epoch awal (epoch 2 dari 8), bukan di akhir training. Itulah sebabnya notebook menyimpan bobot terbaik (`best_state`) berdasarkan test perplexity, bukan bobot epoch terakhir.
- Temperature pada tahap generate mengatur trade-off antara koherensi (temperature rendah → output repetitif) dan keberagaman (temperature tinggi → output lebih acak), sesuai prediksi awal di jurnal.
- `nn.LSTM` memberi hasil sedikit lebih baik dibanding RNN manual (112.9 vs 119.5 test perplexity) dengan biaya ~27% lebih banyak parameter (empat gate vs satu recurrence sederhana), namun kedua arsitektur sama-sama terbatas oleh ukuran data yang kecil.

## Penulis

Nama: Mikail Achmad  
NIM: 24/542370/PA/23026  
Kelas: KOM - B  
Tugas: Pembelajaran Mesin Mendalam | Assignment 3 (RNN Lab: Next-Word Prediction & Poem Generation)
