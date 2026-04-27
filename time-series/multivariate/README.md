# 📊 Household Electric Power Forecasting
Project ini dibuat untuk pembelajaran *Time Series Forecasting* menggunakan Deep Learning.

Project ini bertujuan untuk melakukan **prediksi konsumsi listrik rumah tangga** menggunakan model *Deep Learning* berbasis **TensorFlow (MLP/Dense Neural Network)**. Dataset yang digunakan merupakan data historis konsumsi listrik dengan pendekatan *time series forecasting*.

---

## 📁 Dataset

Dataset diambil dari Google Drive:

* 🔗 https://drive.google.com/uc?id=1AZRfFoyekqSYpri5183RmJjciRGz_ood
---

## ⚙️ Tahapan Project

### 1. Load Data

Dataset dibaca menggunakan **Pandas** dan index diatur sebagai waktu (`datetime`).

---

### 2. Data Preprocessing

Dilakukan normalisasi menggunakan metode **Min-Max Scaling**:

```
x' = (x - min) / (max - min)
```

**Tujuan:**

* Menyamakan skala data
* Mempercepat proses training model

---

### 3. Split Data

Dataset dibagi menjadi:

* **50% Training**
* **50% Validation**

---

### 4. Windowing Dataset

Menggunakan teknik *sliding window*:

* `N_PAST = 24` → data historis (input)
* `N_FUTURE = 24` → data yang diprediksi (output)
* `SHIFT = 1`

**Output:**

* X: 24 timestep sebelumnya
* Y: 24 timestep ke depan

---

### 5. Arsitektur Model

Model menggunakan pendekatan **Multi-Layer Perceptron (MLP)**:

```
Input (24 x N_FEATURES)
↓ Flatten
↓ Dense (64, ReLU)
↓ Dense (32, ReLU)
↓ Dense (24 x N_FEATURES)
↓ Reshape (24, N_FEATURES)
```

---

## 🧠 Training Model

* Optimizer: Adam (`learning_rate = 1e-3`)
* Loss Function: Mean Absolute Error (MAE)
* Metrics: MAE
* Epoch: Maksimal 100

### ⏹️ Early Stopping Custom

Training akan otomatis berhenti jika:

* MAE < 0.055
* Validation MAE < 0.055

---

## 📈 Hasil

Model menghasilkan prediksi dalam bentuk:

```
(batch_size, 24, N_FEATURES)
```

Contoh penggunaan:

```python
train_pred = model.predict(train_set)
print(train_pred[0][0])
```

---

## 🚀 Cara Menjalankan

1. Install dependencies:

```bash
pip install pandas tensorflow
```

2. Jalankan script:

```bash
python household-electric-power.py
```

---

## 📦 Dependencies

* Python 3.x
* pandas
* tensorflow

---

## 💡 Catatan

* Model ini menggunakan **Dense layer**, bukan LSTM/GRU, sehingga input di-*flatten*.
* Cocok sebagai baseline model untuk time series forecasting.
---

## 👨‍💻 Author
**Romualdus Hary Prabowo**

[![Instagram](https://img.shields.io/badge/Instagram-%23E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://www.instagram.com/hyporom._)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/hypo/)
