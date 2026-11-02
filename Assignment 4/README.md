# Transformer Lab: Multi-Head Attention, Debugging & Hugging Face

Implementasi Python (Jupyter Notebook) untuk tugas mata kuliah Pembelajaran Mesin Mendalam. Notebook ini mengimplementasikan **Scaled Dot-Product Attention** dan **Multi-Head Attention** dari nol (`torch.nn.Module`, tanpa modul attention bawaan PyTorch/Hugging Face), mendiagnosis bug arsitektural pada sebuah Transformer encoder layer lewat `forward_hook` dan visualisasi attention weight, lalu membandingkan hasilnya dengan fine-tuning `distilbert-base-uncased` dari Hugging Face untuk tugas yang sama: klasifikasi sentimen review film (NLTK `movie_reviews`).

## Struktur Repo

```
Assignment 4/
├── README.md
├── transformer_assignment.ipynb   # notebook final (Task 0 + Part A-D terisi, sudah dijalankan)
└── results/                       # checkpoint Trainer Hugging Face (di-gitignore, dibuat ulang otomatis saat notebook dijalankan)
```

## Cara menjalankan

Notebook dijalankan secara lokal (Jupyter/VSCode) dengan Python 3.12:

1. Install dependency: `pip install torch transformers datasets nltk matplotlib` (baris `!pip install` pertama di cell **Setup** juga sudah menangani ini).
2. Buka `transformer_assignment.ipynb`, jalankan seluruh cell dari atas ke bawah (**Restart Kernel and Run All**). Dataset `movie_reviews` otomatis diunduh lewat `nltk.download`, dan tokenizer/model `distilbert-base-uncased` otomatis diunduh dari Hugging Face Hub saat Part C dijalankan.
3. Training di Part C (Hugging Face) berjalan di **CPU** secara default (`torch.cuda.is_available()` → `False` di environment pengerjaan ini), jadi `batch_size` dikecilkan ke 8 dan `max_length` token ke 128 supaya tetap bisa jalan dengan RAM terbatas tanpa OOM.

## Struktur Notebook

### Task 0 | Warm-up & Reflection

Refleksi satu paragraf tentang ide inti self-attention (setiap kata menjadi *query* yang membandingkan dirinya ke semua kata lain sebagai *key*, lalu mengambil informasi dari *value* secara berbobot) dan kenapa ini mengatasi *bottleneck* RNN/LSTM: RNN butuh `O(sequence length)` langkah untuk dua kata yang berjauhan posisinya saling memengaruhi (lambat, rawan *vanishing gradient*, tidak bisa diparalelkan), sementara self-attention membuat semua kata berinteraksi langsung dalam `O(1)` langkah.

### Part A | `MultiHeadAttention` dari Nol

- `scaled_dot_product_attention(Q, K, V, mask)`: `softmax((QKᵀ)/√d_k + mask) · V`, dengan posisi yang di-mask diisi `-1e9` sebelum softmax.
- `MultiHeadAttention`: 4 linear layer (`W_q, W_k, W_v, W_o`), input diproyeksi ke Q/K/V, dipecah ke beberapa head (`split_heads`), dilewatkan ke fungsi di atas, digabung kembali (concat heads), lalu diproyeksikan lewat `W_o`.
- Divalidasi lewat unit test shape: input `(4, 10, 64)` → context vector `(4, 10, 64)`, attention weights `(4, 8, 10, 10)` — **lolos**.

### Part B | Debugging Lewat Attention Weights & Hooks

`BuggyTransformerEncoderLayer` (disediakan, memakai `MultiHeadAttention` dari Part A) gagal belajar — akurasi klasifikasi mentok di **~52-53%** (setara menebak) setelah 3 epoch. Dengan memasang `register_forward_hook` pada modul `MultiHeadAttention` untuk menyadap attention weight selama training, lalu memvisualisasikannya sebagai heatmap, teridentifikasi bahwa attention weight-nya **acak/tidak fokus** (tidak ada pola diagonal maupun blok kolom/baris yang konsisten).

Root cause: residual connection yang hilang di sub-layer self-attention —

```python
attn_output, attn_weights = self.mha(x, mask)
x = self.dropout1(attn_output)   # <- menimpa x, input asli sebelum attention hilang
x = self.norm1(x)                # <- harusnya self.norm1(x + self.dropout1(attn_output))
```

Tanpa residual connection, gradien yang mengalir balik ke parameter attention (`W_q, W_k, W_v`) menjadi sangat lemah (loss landscape jauh lebih kasar tanpa *skip connection*), sehingga attention tidak pernah belajar pola yang bermakna.

### Part C | Fine-tuning `distilbert-base-uncased` (Hugging Face)

Dataset NLTK `movie_reviews` yang sama (1800 train / 200 test) di-fine-tune ke `distilbert-base-uncased` lewat `transformers.Trainer` (1 epoch, `max_length=128`, `batch_size=8`). **Akurasi akhir: 70.00%** — jauh di atas model buggy di Part B, menunjukkan betapa besar dampak satu baris residual connection yang hilang terhadap kemampuan belajar model.

### Part D | Refleksi Akhir

Membandingkan pengalaman *building* (Part A) vs *debugging* (Part B), menghubungkan `MultiHeadAttention` buatan sendiri dengan arsitektur DistilBERT (6 layer × 12 head), dan merefleksikan nilai memahami komponen dari dasar meski di praktiknya hampir selalu memakai library seperti Hugging Face.

## Hasil

| Eksperimen                                   | Akurasi Test |
| --------------------------------------------- | ------------ |
| `BuggyClassifier` (residual connection hilang) | ~52-53%      |
| `distilbert-base-uncased` fine-tuned (Hugging Face) | **70.00%**   |

## Kesimpulan

- `MultiHeadAttention` buatan sendiri tervalidasi benar secara matematis (lolos unit test shape) dan secara fungsional (dipakai di `BuggyTransformerEncoderLayer` tanpa error runtime) — bukti bahwa komponen attention bukan sumber masalah.
- Satu baris residual connection yang hilang sudah cukup membuat seluruh model gagal belajar (mentok ~52-53%, setara menebak), membuktikan betapa krusialnya *skip connection* untuk aliran gradien di arsitektur Transformer yang dalam — konsisten dengan materi kuliah soal *loss landscape* yang jauh lebih kasar tanpa residual connection.
- `forward_hook` PyTorch terbukti efektif sebagai alat debugging non-invasif: attention weight bisa disadap tanpa mengubah kode model sama sekali, dan visualisasinya langsung mengungkap bahwa attention gagal belajar pola apa pun (acak, bukan fokus ke token tertentu).
- Fine-tuning `distilbert-base-uncased` (70% akurasi dengan ~5 baris kode inti: load tokenizer, load model, tokenize, `Trainer`, `.train()`) menunjukkan nilai abstraksi Hugging Face dibanding membangun seluruh Transformer encoder dari nol — tapi memahami isi `MultiHeadAttention` dari Part A adalah yang membuat bug di Part B bisa ditemukan dan dijelaskan secara tepat.

## Penulis

Nama: Mikail Achmad  
NIM: 24/542370/PA/23026  
Kelas: KOM - B  
Tugas: Pembelajaran Mesin Mendalam | Assignment 4 (Transformer: Multi-Head Attention, Debugging & Hugging Face)
