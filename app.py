import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model dan vectorizer
with open('spam_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Judul aplikasi
st.title("📧 Spam Email Detector")

# Buat tabs — navigasi seperti menu
tab1, tab2 = st.tabs(["🔍 Cek Spam", "📊 Dashboard"])

# ================================
# TAB 1 — CEK SPAM
# ================================
with tab1:
    st.subheader("Masukkan pesan untuk dicek apakah spam atau bukan")
    
    pesan = st.text_area("✉️ Tulis pesanmu di sini:", height=150)
    
    if st.button("🔍 Cek Sekarang"):
        if pesan.strip() == "":
            st.warning("⚠️ Pesan tidak boleh kosong!")
        else:
            pesan_tfidf = tfidf.transform([pesan])
            hasil = model.predict(pesan_tfidf)[0]
            proba = model.predict_proba(pesan_tfidf)[0]
            
            if hasil == 1:
                st.error("🚨 SPAM! Pesan ini terdeteksi sebagai spam!")
            else:
                st.success("✅ BUKAN SPAM! Pesan ini aman.")
            
            # Tampilkan confidence score
            st.write("---")
            st.write("**Tingkat kepercayaan model:**")
            col1, col2 = st.columns(2)
            col1.metric("Bukan Spam", f"{proba[0]*100:.1f}%")
            col2.metric("Spam", f"{proba[1]*100:.1f}%")

# ================================
# TAB 2 — DASHBOARD
# ================================
with tab2:
    st.subheader("📊 Dashboard Analisis Dataset")
    
    # Load dataset
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    df['panjang_pesan'] = df['message'].apply(len)
    
    # Metric cards
    st.write("### 📈 Ringkasan Dataset")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pesan", len(df))
    col2.metric("Spam", len(df[df['label']=='spam']))
    col3.metric("Ham", len(df[df['label']=='ham']))
    
    st.write("---")
    
    # Grafik 1 — Distribusi spam vs ham
    st.write("### 📊 Distribusi Spam vs Ham")
    fig1, ax1 = plt.subplots(figsize=(6,4))
    sns.countplot(x='label', data=df, hue='label', palette='Set2', legend=False, ax=ax1)
    ax1.set_title('Jumlah Spam vs Ham')
    ax1.set_xlabel('Label')
    ax1.set_ylabel('Jumlah')
    st.pyplot(fig1)
    
    st.write("---")
    
    # Grafik 2 — Panjang pesan
    st.write("### 📏 Perbandingan Panjang Pesan")
    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.histplot(data=df, x='panjang_pesan', hue='label', bins=50, palette='Set2', ax=ax2)
    ax2.set_title('Panjang Pesan Spam vs Ham')
    ax2.set_xlabel('Panjang Pesan (karakter)')
    ax2.set_ylabel('Jumlah')
    st.pyplot(fig2)
    
    st.write("---")
    
    # Tabel statistik
    st.write("### 📋 Statistik Panjang Pesan")
    stats = df.groupby('label')['panjang_pesan'].agg(['mean', 'min', 'max'])
    stats.columns = ['Rata-rata', 'Minimum', 'Maksimum']
    stats = stats.round(1)
    st.dataframe(stats)
