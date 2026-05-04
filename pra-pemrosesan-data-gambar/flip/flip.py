import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Membaca citra
image_path = "../data/singa.jpg"
image = Image.open(image_path)

# Menampilkan citra asli
plt.figure(figsize=(8, 4)) # Membuat kanvas dengan ukuran panjang 8 dan lebar 4
plt.subplot(1, 3, 1) # Membuat subplot pertama dari 3 subplot dalam satu baris
plt.title("Original Image") # Memberikan judul pada subplot pertama
plt.imshow(image) # Menampilkan citra asli pada subplot pertama
plt.axis('off') # Menghilangkan sumbu (x, y) pada subplot pertama

# Transformasi untuk flip vertikal
vertical_flip = transforms.Compose([ # Membuat transformasi yang terdiri dari satu langkah yaitu flip vertikal
 transforms.RandomVerticalFlip(p=1) # p=1 berarti flip vertikal akan selalu dilakukan (100% probabilitas)
])
flipped_image_vertical = vertical_flip(image) # Menerapkan transformasi flip vertikal pada citra asli

# Menampilkan citra setelah flip vertikal
plt.subplot(1, 3, 2) # Membuat subplot kedua dari 3 subplot dalam satu baris
plt.title("Vertical Flip") # Memberikan judul pada subplot kedua
plt.imshow(flipped_image_vertical) # Menampilkan citra setelah flip vertikal pada subplot kedua
plt.axis('off') # Menghilangkan sumbu (x, y) pada subplot kedua
plt.title("Vertical Flip") # Memberikan judul pada subplot kedua
plt.imshow(flipped_image_vertical) # Menampilkan citra setelah flip vertikal pada subplot kedua
plt.axis('off') # Menghilangkan sumbu (x, y) pada subplot kedua

# Transformasi untuk flip horizontal
horizontal_flip = transforms.Compose([ # Membuat transformasi yang terdiri dari satu langkah yaitu flip horizontal
 transforms.RandomHorizontalFlip(p=1) # p=1 berarti flip horizontal akan selalu dilakukan (100% probabilitas)
])
flipped_image_horizontal = horizontal_flip(image) # Menerapkan transformasi flip horizontal pada citra asli

# Menampilkan citra setelah flip horizontal
plt.subplot(1, 3, 3) # Membuat subplot ketiga dari 3 subplot dalam satu baris
plt.title("Horizontal Flip")  # Memberikan judul pada subplot ketiga
plt.imshow(flipped_image_horizontal) # Menampilkan citra setelah flip horizontal pada subplot ketiga
plt.axis('off') # Menghilangkan sumbu (x, y) pada subplot ketiga

plt.tight_layout() # Mengatur tata letak subplot agar tidak saling tumpang tindih
plt.show() # Menampilkan semua subplot ke layar