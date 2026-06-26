import pandas as pd
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) 
nltk.download('stopwords', quiet=True)

print("--- Memuat Dataset Mentah ---")
df = pd.read_csv('dataset_mentah.csv')

factory = StemmerFactory()
stemmer = factory.create_stemmer()

list_stopwords = set(stopwords.words('indonesian'))
list_stopwords.update(['yg', 'nya', 'sih', 'di', 'ke', 'dari', 'buat', 'sama', 'kok', 'deh', 'aja', 'utk', 'udah', 'dlm', 'kalo'])

kata_negasi = {'tidak', 'bukan', 'belum', 'jangan', 'kurang', 'enggak', 'ga', 'gak', 'tak'}
list_stopwords = list_stopwords - kata_negasi

def bersihkan_teks(text):
    text = str(text).lower() 
    text = re.sub(r'http\S+|www\S+', '', text) 
    text = re.sub(r'[^a-z\s]', '', text) 
    text = re.sub(r'\s+', ' ', text).strip() 
    
    tokens = word_tokenize(text) 
    tokens = [kata for kata in tokens if kata not in list_stopwords] 
    
    # 1. Stemming terlebih dahulu
    stemmed_tokens = [stemmer.stem(kata) for kata in tokens] 
    
    # 2. TRIK PAKSA NEGASI: Gabungkan kata negasi dengan kata setelahnya
    final_tokens = []
    i = 0
    while i < len(stemmed_tokens):
        if stemmed_tokens[i] in kata_negasi and i + 1 < len(stemmed_tokens):
            # Gabungkan menjadi satu kata tanpa spasi (misal: "tidaksuka")
            gabungan = stemmed_tokens[i] + stemmed_tokens[i+1]
            final_tokens.append(gabungan)
            i += 2 # Lompat 2 kata karena sudah digabung
        else:
            final_tokens.append(stemmed_tokens[i])
            i += 1
            
    teks_bersih = " ".join(final_tokens) 
    return teks_bersih

print("\nMemulai proses pembersihan teks KHUSUS NEGASI (Re-run Sastrawi)...")
tqdm.pandas(desc="Proses Preprocessing")

df['teks_bersih'] = df['teks_ulasan'].progress_apply(bersihkan_teks)

df = df.dropna(subset=['teks_bersih'])
df = df[df['teks_bersih'] != '']

df.to_csv('dataset_bersih.csv', index=False, encoding='utf-8')
print("\nSukses! Dataset bersih TERBARU telah disimpan.")