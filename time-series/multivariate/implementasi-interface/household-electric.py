import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Forecasting App", layout="wide")
st.title("Aplikasi Prediksi Time Series 📈")
st.write("Memprediksi 24 langkah ke depan berdasarkan data historis terakhir.")

# ==========================================
# 2. FUNGSI LOAD & CACHE
# ==========================================
# st.cache_resource memastikan model hanya di-load sekali saat web pertama kali dibuka
@st.cache_resource
def init_model():
    return tf.keras.models.load_model('../model.h5', compile=False)

# st.cache_data memastikan data tidak di-download ulang setiap kali tombol diklik
@st.cache_data
def load_data():
    df = pd.read_csv(
        "https://drive.google.com/uc?id=1AZRfFoyekqSYpri5183RmJjciRGz_ood",
        sep=",",
        index_col="datetime",
        header=0,
    )
    return df

model = init_model()
df = load_data()
N_FEATURES = len(df.columns)
N_PAST = 24

# ==========================================
# 3. FUNGSI NORMALISASI & DENORMALISASI
# ==========================================
def normalize_series(data, min_val, max_val):
    return (data - min_val) / (max_val - min_val)

def denormalize_series(data_norm, min_val, max_val):
    return data_norm * (max_val - min_val) + min_val

# Hitung nilai min dan max dari keseluruhan data historis
data_mentah = df.values
min_val = data_mentah.min(axis=0)
max_val = data_mentah.max(axis=0)

# ==========================================
# 4. TAMPILAN ANTARMUKA (UI)
# ==========================================
st.subheader("Data Historis Terakhir (5 Baris)")
st.dataframe(df.tail(5)) # Menampilkan data paling ujung dari database

# Tombol untuk memicu prediksi
if st.button("🔮 Lakukan Prediksi 24 Langkah ke Depan", type="primary"):
    
    with st.spinner("Sedang memproses prediksi oleh AI..."):
        # A. Siapkan Data (Ambil 24 langkah terakhir & Normalisasi)
        data_norm = normalize_series(data_mentah, min_val, max_val)
        data_terakhir = data_norm[-N_PAST:] 
        
        # B. Reshape (Membungkus ke dalam dimensi Batch = 1)
        input_prediksi = np.reshape(data_terakhir, (1, N_PAST, N_FEATURES))
        
        # C. Lakukan Prediksi
        hasil_prediksi_norm = model.predict(input_prediksi)
        
        # D. Kembalikan nilai prediksi ke skala asli (Denormalisasi)
        hasil_prediksi_asli = denormalize_series(hasil_prediksi_norm[0], min_val, max_val)
        
        st.success("Prediksi Berhasil!")
        
        # ==========================================
        # 5. VISUALISASI HASIL PREDIKSI
        # ==========================================
        # Membuat DataFrame baru khusus untuk hasil prediksi
        kolom_fitur = df.columns
        df_prediksi = pd.DataFrame(hasil_prediksi_asli, columns=kolom_fitur)
        
        # Membuat penamaan index (Step +1, Step +2, dst)
        df_prediksi.index = [f"Step +{i+1}" for i in range(24)]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**Tabel Angka Prediksi:**")
            st.dataframe(df_prediksi)
            
        with col2:
            st.write("**Grafik Pergerakan Masa Depan:**")
            # fitur line_chart bawaan streamlit untuk visualisasi sederhana
            st.line_chart(df_prediksi)