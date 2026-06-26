import streamlit as st
import joblib
import re
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

st.set_page_config(page_title="Analisis Sentimen IndoNLU", page_icon="🤖", layout="centered")

@st.cache_resource
def load_resources():
    model = joblib.load('model_sentiment.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    
    list_stopwords = set(stopwords.words('indonesian'))
    list_stopwords.update(['yg', 'nya', 'sih', 'di', 'ke', 'dari', 'buat', 'sama', 'kok', 'deh', 'aja', 'utk', 'udah', 'dlm', 'kalo'])
    
    kata_negasi = {'tidak', 'bukan', 'belum', 'jangan', 'kurang', 'enggak', 'ga', 'gak', 'tak'}
    list_stopwords = list_stopwords - kata_negasi
    
    return model, vectorizer, stemmer, list_stopwords, kata_negasi

model, vectorizer, stemmer, list_stopwords, kata_negasi = load_resources()

def bersihkan_teks(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = word_tokenize(text)
    tokens = [kata for kata in tokens if kata not in list_stopwords]
    
    stemmed_tokens = [stemmer.stem(kata) for kata in tokens]
    
    # TRIK PAKSA NEGASI
    final_tokens = []
    i = 0
    while i < len(stemmed_tokens):
        if stemmed_tokens[i] in kata_negasi and i + 1 < len(stemmed_tokens):
            gabungan = stemmed_tokens[i] + stemmed_tokens[i+1]
            final_tokens.append(gabungan)
            i += 2
        else:
            final_tokens.append(stemmed_tokens[i])
            i += 1
            
    teks_bersih = " ".join(final_tokens)
    return teks_bersih

st.title("🤖 Deteksi Sentimen Teks Indonesia ")
st.divider()

with st.form(key='nlp_form'):
    input_teks = st.text_area("Masukkan teks di sini:", height=150)
    submit_button = st.form_submit_button(label='Analisis Sentimen')

if submit_button:
    if input_teks.strip() == "":
        st.warning("Mohon masukkan teks terlebih dahulu!")
    else:
        with st.spinner("Menganalisis..."):
            teks_bersih = bersihkan_teks(input_teks)
            teks_vektor = vectorizer.transform([teks_bersih])
            prediksi = model.predict(teks_vektor)[0]
            
            st.subheader("Hasil Analisis:")
            if prediksi == 'positive':
                st.success("🟢 Sentimen: **POSITIF**")
            elif prediksi == 'negative':
                st.error("🔴 Sentimen: **NEGATIF**")
            else:
                st.info("⚪ Sentimen: **NETRAL**")
                
            with st.expander("Lihat detail pra-pemrosesan"):
                st.write("**Teks Asli:**", input_teks)
                st.write("**Teks Bersih (Kata digabung mutlak):**", teks_bersih)