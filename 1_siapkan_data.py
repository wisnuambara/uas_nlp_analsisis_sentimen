import pandas as pd
import requests
import io

print("--- Mengunduh Dataset IndoNLU ---")
# Menggunakan dataset IndoNLU (Sentiment Analysis) langsung dari GitHub resminya
url = "https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv"

# Mengambil data menggunakan requests
response = requests.get(url)
response.raise_for_status() # Memastikan unduhan berhasil

# Membaca data TSV (Tab Separated Values) ke dalam Pandas DataFrame
df = pd.read_csv(io.StringIO(response.text), sep='\t', header=None, names=['teks_ulasan', 'label'])

# ==========================================
# PEMAHAMAN DATA (Untuk Dokumentasi README.md)
# ==========================================
print("\nKarakteristik Data & Distribusi Label:")
print(df['label'].value_counts())
print(f"Total Data: {len(df)} baris")

# Menyimpan ke format CSV lokal agar lebih mudah diproses di tahap selanjutnya
df.to_csv('dataset_mentah.csv', index=False, encoding='utf-8')
print("\nSukses! Data telah disimpan sebagai 'dataset_mentah.csv'")