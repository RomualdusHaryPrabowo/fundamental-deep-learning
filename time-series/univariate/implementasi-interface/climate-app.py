import streamlit as st
import numpy as np
import tensorflow as tf
import requests
from datetime import date, timedelta
import pandas as pd

# ---------------------------------------------------------
# BAGIAN 1: PENGATURAN HALAMAN WEB
# ---------------------------------------------------------
st.set_page_config(page_title="Prediksi Suhu AI", page_icon="🌡️")
st.title("Aplikasi Prediksi Suhu Berbasis AI")
st.write("Aplikasi ini secara otomatis menarik data suhu rata-rata 60 hari terakhir di **Kota Metro, Lampung**.")
st.write("Dengan menggunakan model AI yang telah dilatih, aplikasi ini memprediksi suhu rata-rata untuk esok hari berdasarkan pola data masa lalu.")
st.markdown("---")

# ---------------------------------------------------------
# BAGIAN 2: MEMUAT MODEL AI
# ---------------------------------------------------------
# st.cache_resource memastikan model hanya di-load 1 kali untuk menghemat memori
@st.cache_resource
def load_my_model():
    try:
        return tf.keras.models.load_model('../model.h5')
    except Exception as e:
        st.error(f"Gagal memuat model. Pastikan file 'model.h5' ada di folder ini. Error: {e}")
        return None

model = load_my_model()

# ---------------------------------------------------------
# BAGIAN 3: FUNGSI MENGAMBIL DATA API (OPEN-METEO)
# ---------------------------------------------------------
def get_60_days_weather():
    # Menghitung tanggal: butuh data dari 60 hari yang lalu sampai hari ini
    end_date = date.today()
    start_date = end_date - timedelta(days=59) # Total 60 hari termasuk hari ini
    
    # Format tanggal ke string (YYYY-MM-DD)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    # Koordinat Kota Metro, Lampung (Latitude, Longitude)
    lat = "-5.1131"
    lon = "105.3067"
    
    # URL API Open-Meteo
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_str}&end_date={end_str}&daily=temperature_2m_mean&timezone=Asia%2FJakarta"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        suhu_harian = data['daily']['temperature_2m_mean']
        tanggal = data['daily']['time']
        return suhu_harian, tanggal
    else:
        st.error("Gagal mengambil data dari API Open-Meteo.")
        return None, None

# ---------------------------------------------------------
# BAGIAN 4: LOGIKA UTAMA (TOMBOL & PREDIKSI)
# ---------------------------------------------------------
if st.button("Tarik Data BMKG/Satelit & Prediksi Suhu Besok", type="primary"):
    if model is not None:
        with st.spinner('Sedang menghubungi satelit cuaca & AI sedang berpikir...'):
            
            # 1. Mengambil data dari API
            suhu_60_hari, daftar_tanggal = get_60_days_weather()
            
            if suhu_60_hari is not None:
                # 2. Menampilkan grafik data 60 hari terakhir ke layar
                st.subheader("📊 Data Suhu 60 Hari Terakhir (Kota Metro)")
                df_cuaca = pd.DataFrame({
                    "Tanggal": pd.to_datetime(daftar_tanggal),
                    "Suhu (°C)": suhu_60_hari
                }).set_index("Tanggal")
                
                st.line_chart(df_cuaca) # Menampilkan grafik interaktif
                
                # 3. PREPROCESSING: Mengubah data menjadi bentuk 3D untuk LSTM
                # Jika ada data yang kosong (null) dari API, kita isi dengan rata-rata sementara
                suhu_bersih = [s if s is not None else 30.0 for s in suhu_60_hari] 
                data_input = np.array(suhu_bersih)
                data_input = np.reshape(data_input, (1, 60, 1)) # Format: (Batch, Window_size, Feature)
                
                # 4. PREDIKSI MENGGUNAKAN MODEL
                hasil_prediksi = model.predict(data_input)
                suhu_besok = hasil_prediksi[0][0]
                
                # 5. MENAMPILKAN HASIL AKHIR
                st.markdown("---")
                st.subheader("🎯Kesimpulan AI")
                st.success(f"Berdasarkan pola 60 hari di atas, prediksi rata-rata suhu untuk esok hari adalah **{suhu_besok:.2f} °C**")