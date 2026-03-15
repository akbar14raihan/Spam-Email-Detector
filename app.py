import streamlit as st
import pickle

# Load model dan vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Tampilan aplikasi
st.title("📧 Spam Email Detector")
st.subheader("Masukkan pesan untuk dicek apakah spam atau bukan")

# Input teks dari user
pesan = st.text_area("✉️ Tulis pesanmu di sini:", height=150)

# Tombol prediksi
if st.button("🔍 Cek Sekarang"):
    if pesan.strip() == "":
        st.warning("⚠️ Pesan tidak boleh kosong!")
    else:
        # Proses prediksi
        pesan_tfidf = tfidf.transform([pesan])
        hasil = model.predict(pesan_tfidf)[0]
        
        # Tampilkan hasil
        if hasil == 1:
            st.error("🚨 SPAM! Pesan ini terdeteksi sebagai spam!")
        else:
            st.success("✅ BUKAN SPAM! Pesan ini aman.")