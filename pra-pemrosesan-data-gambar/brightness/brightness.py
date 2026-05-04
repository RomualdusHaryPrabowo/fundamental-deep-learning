import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# load image
image_path = "../data/singa.jpg"
image = Image.open(image_path)

# Membuat kanvas visualisasi (lebar 10, tinggi 4)
plt.figure(figsize=(10, 4))

# Menempatkan gambar asli di posisi 1
plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

# ==========================================================
# 2. PROSES TRANSFORMASI KECERAHAN (COLOR JITTER)
# Keterangan: Nilai 1.0 adalah kecerahan asli gambar.
# ==========================================================

# A. Kecerahan Acak Dua Arah (Gelap/Terang)
# Memasukkan 1 nilai (0.5) akan membuat rentang [1.0 - 0.5, 1.0 + 0.5] -> [0.5, 1.5].
# Hasil: Gambar diacak menjadi maksimal 50% lebih gelap ATAU maksimal 50% lebih terang.
random_brightness = transforms.Compose([
    transforms.ColorJitter(brightness=0.5) 
])
random_image = random_brightness(image)


# B. Kecerahan Spesifik (Hanya Lebih Terang)
# Memasukkan tuple (1, 1.5) mengunci rentang dari batas asli (1.0) hingga +50% terang (1.5).
# Hasil: Gambar pasti menjadi lebih terang atau maksimal sama seperti aslinya.
brightness_adjustment = transforms.Compose([
    transforms.ColorJitter(brightness=(1, 1.5)) 
])
brightened_image = brightness_adjustment(image)


# C. Kecerahan Spesifik (Hanya Lebih Gelap)
# Memasukkan tuple (0.5, 1) mengunci rentang dari -50% gelap (0.5) hingga batas asli (1.0).
# Hasil: Gambar pasti menjadi lebih gelap atau maksimal sama seperti aslinya.
darkness_adjustment = transforms.Compose([
    transforms.ColorJitter(brightness=(0.5, 1)) 
])
darkened_image = darkness_adjustment(image)

# ==========================================================
# 3. MENAMPILKAN HASIL TRANSFORMASI
# ==========================================================

# Menempatkan gambar acak di posisi 2
plt.subplot(1, 4, 2)
plt.title("Random Brightness")
plt.imshow(random_image)
plt.axis('off')

# Menempatkan gambar yang diterangkan di posisi 3
plt.subplot(1, 4, 3)
plt.title("Brightened (1.0 to 1.5)")
plt.imshow(brightened_image)
plt.axis('off')

# Menempatkan gambar yang digelapkan di posisi 4
plt.subplot(1, 4, 4)
plt.title("Darkened (0.5 to 1.0)")
plt.imshow(darkened_image)
plt.axis('off')

# Merapikan jarak antar gambar dan menampilkannya ke layar
plt.tight_layout()
plt.show()