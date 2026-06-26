import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("--- 1. Memuat Dataset Bersih Terbaru ---")
df = pd.read_csv('dataset_bersih.csv')
df = df.dropna(subset=['teks_bersih'])

X = df['teks_bersih']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("--- 2. Mengekstraksi Fitur ---")
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("--- 3. Melatih Model ---")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("--- 4. Evaluasi Model ---")
y_pred = model.predict(X_test_tfidf)
akurasi = accuracy_score(y_test, y_pred)
print(f"Akurasi Model Baru: {akurasi * 100:.2f}%\n")

joblib.dump(model, 'model_sentiment.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("Model dan Vectorizer berhasil diperbarui dengan fitur anti-negasi.")