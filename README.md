# 🤖 Sistem Deteksi Sentimen Teks Indonesia (IndoNLU)

Sistem berbasis *Natural Language Processing* (NLP) ini dibangun untuk menganalisis sentimen teks berbahasa Indonesia (Positif, Negatif, Netral). Model dilatih menggunakan algoritma Naive Bayes dan dilengkapi dengan fitur **Negation Handling** untuk mengatasi kelemahan sentimen pada kata negasi (seperti "tidak suka" atau "kurang baik").

## 📊 1. Dataset & Pemahaman Data
Proyek ini menggunakan dataset publik **IndoNLU (Indonesian Natural Language Understanding)** khusus pada domain *Sentiment Analysis* (SMSA). 
* **Karakteristik Data:** Teks bersumber dari ulasan, komentar sosial media, dan teks umum berbahasa Indonesia.
* **Total Data:** ~11.000 baris.
* **Distribusi Label:** Didominasi oleh kelas Positif, diikuti Negatif, dan Netral.

## ⚙️ 2. Pra-pemrosesan Teks & Ekstraksi Fitur
Pembersihan teks menggunakan kombinasi *Regular Expression*, pustaka **NLTK**, dan **Sastrawi**. Tahapan meliputi:
1.  **Case Folding & Cleansing:** Mengubah ke huruf kecil dan menghapus URL/tanda baca.
2.  **Stopword Removal (Modifikasi):** Menghapus kata hubung tak bermakna, **KECUALI** kata negasi (tidak, bukan, belum, dll) untuk menjaga konteks kalimat.
3.  **Stemming:** Mengubah kata ke bentuk dasar menggunakan `Sastrawi`.
4.  **Negation Handling (Token Concatenation):** Menggabungkan kata negasi dengan kata setelahnya (contoh: `tidak` + `suka` -> `tidaksuka`) agar model Naive Bayes tidak terkecoh oleh probabilitas kata positif.
5.  **Ekstraksi Fitur:** Menggunakan metode representasi numerik **TF-IDF**.

## 🧠 3. Pelatihan & Evaluasi Model
Model dilatih menggunakan algoritma klasifikasi **Multinomial Naive Bayes**.
* **Pembagian Data:** 80% Data Latih (Train) & 20% Data Uji (Test).
* **Akurasi Model:** ~80%
* Model sangat baik dalam mengenali sentimen positif dan negatif murni, dan telah dioptimasi agar tidak terjadi *False Positive* pada kalimat negasi.

## 🚀 4. Cara Menjalankan Aplikasi Lokal
Aplikasi antarmuka dikembangkan menggunakan **Streamlit**.

1. Pastikan Python sudah terinstal, lalu *clone* repository ini.
2. Buka terminal di folder proyek, buat *virtual environment*, dan instal dependensi:
   ```bash
   pip install -r requirements.txt