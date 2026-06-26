# 🤖 Sistem Deteksi Sentimen Teks Indonesia (IndoNLU)

Repositori ini berisi proyek akhir (UAS) Natural Language Processing (NLP) berupa sistem klasifikasi sentimen teks berbahasa Indonesia. Sistem ini dapat menerima input teks dari pengguna dan mengklasifikasikannya ke dalam tiga kategori: **Positif, Negatif, atau Netral**.

Proyek ini dibangun secara utuh mulai dari pemrosesan data mentah hingga *deployment* ke dalam bentuk aplikasi web interaktif.

---

## 📑 Tahap 1: Membangun Model NLP

### 1. Pengumpulan & Pemahaman Data
Dataset yang digunakan berasal dari standar publik **IndoNLU (Indonesian Natural Language Understanding)** khusus domain *Sentiment Analysis* (SMSA).
* **Karakteristik Data:** Berisi kalimat bahasa Indonesia sehari-hari dari ulasan, opini, dan media sosial. Bahasa yang digunakan mencakup bahasa baku dan *slang*.
* **Distribusi Label:** Total ~11.000 data ulasan. Label didominasi oleh kelas **Positif** (60%), kemudian **Negatif** (30%), dan **Netral** (10%). Dataset ini merepresentasikan kondisi riil data teks yang sering kali tidak seimbang (*imbalanced*).

### 2. Pra-pemrosesan Teks (Pustaka: NLTK & Sastrawi)
Teks dibersihkan melalui beberapa tahapan ketat sebelum diekstraksi:
* **Case Folding & Cleansing:** Mengubah teks menjadi huruf kecil (*lowercase*) serta menghapus URL, angka, dan karakter selain huruf (tanda baca).
* **Tokenisasi & Stopword Removal (Dengan Modifikasi):** Memecah kalimat dan menghapus kata hubung (menggunakan `nltk.corpus.stopwords`). 
  > **Trik Khusus (Negation Handling):** Kata-kata negasi (seperti *tidak, bukan, belum, jangan*) dikecualikan dari penghapusan *stopword*.
* **Stemming (Sastrawi):** Mengembalikan kata berimbuhan ke bentuk dasarnya (misal: "pelayanannya" -> "layan").
* **Token Concatenation (Anti-Negasi):** Untuk mengatasi kelemahan model dalam membaca konteks, kata negasi secara otomatis digabungkan dengan kata setelahnya (contoh: "tidak" + "suka" menjadi satu *token* unik **"tidaksuka"**). Ini mencegah model keliru memberikan sentimen positif pada kalimat penolakan.

### 3. Ekstraksi Fitur / Representasi Teks
Menggunakan algoritma **TF-IDF (Term Frequency-Inverse Document Frequency)** melalui pustaka `scikit-learn` (`TfidfVectorizer`). Representasi diatur menggunakan **Unigram dan Bigram (N-gram 1,2)** untuk menangkap konteks gabungan kata dengan lebih akurat.

### 4. Pelatihan Model
* **Algoritma:** Multinomial Naive Bayes (`MultinomialNB`).
* **Pembagian Data:** Dataset dipecah menjadi **80% Data Latih** (Train) dan **20% Data Uji** (Test) menggunakan `train_test_split` dengan `random_state=42`.

### 5. Evaluasi Model
Berdasarkan hasil pengujian pada 20% data uji, model menunjukkan performa yang sangat solid:
* **Accuracy:** ~80%
* **Precision, Recall, F1-Score:** Model sangat presisi dalam menebak kelas Positif (F1-score ~0.87) dan Negatif (F1-score ~0.71). Kelas Netral memiliki tantangan tersendiri karena ketiadaan fitur emosional yang kuat dan jumlah data yang lebih sedikit.
* **Analisis Confusion Matrix:** Kesalahan klasifikasi (*False Positives*) berhasil ditekan secara signifikan setelah penerapan teknik *Token Concatenation* pada kata negasi, membuktikan bahwa pra-pemrosesan yang baik dapat menutupi kelemahan asumsi independensi pada algoritma Naive Bayes.

---

## 💻 Tahap 2: Mengaplikasikan Model

### 1. Penyimpanan Model
Setelah dilatih, model klasifikasi (`model_sentiment.pkl`) dan komponen ekstraksi fitur (`tfidf_vectorizer.pkl`) disimpan menggunakan modul `joblib`. Hal ini memastikan model dapat dimuat ulang oleh aplikasi tanpa perlu melalui tahap *training* komputasi yang berat.

### 2. Antarmuka Pengguna & Penanganan Input
Aplikasi interaktif dibangun menggunakan **Streamlit**. 
* Terdapat *Text Area* agar pengguna dapat memasukkan teks bebas.
* **Penanganan Input:** Teks dari pengguna dilewatkan ke fungsi pra-pemrosesan yang **sama persis** dengan saat tahap pelatihan (termasuk *stemming* Sastrawi dan penyatuan kata negasi) sebelum diubah oleh Vectorizer dan diprediksi oleh Model.
* Hasil akhir akan memunculkan *badge* sentimen beserta rincian teks mentah (*Pre-processed Text*) sebagai bentuk transparansi proses.

---

## 🚀 Cara Menjalankan Proyek di Komputer Lokal

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/wisnuambara/uas_nlp_analsisis_sentimen.git
   cd [yourrepositoryname]
   python -m venv env
   env\Scripts\activate
   pip install -r requirements.txt
   streamlit run 4_app.py