import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Membaca citra
image_path = "../data/singa.jpg"
image = Image.open(image_path)

# Menampilkan citra asli
plt.figure(figsize=(8, 4))
plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

# Transformasi untuk rotasi sebesar 30 derajat searah jarum jam
rotation = transforms.Compose([
 transforms.RandomRotation(degrees=30)
])
rotated_image = rotation(image)

# Menampilkan citra setelah rotasi
plt.subplot(1, 3, 2)
plt.title("Rotated Image")
plt.imshow(rotated_image)
plt.axis('off')

# Transformasi untuk rotasi sebesar 30 derajat berlawanan arah jarum jam
inverse_rotation = transforms.Compose([
 transforms.RandomRotation(degrees=(-30, 0))
])
inverse_rotated_image = inverse_rotation(image)

# Menampilkan citra setelah rotasi berlawanan arah jarum jam
plt.subplot(1, 3, 3)
plt.title("Inverse Rotated Image")
plt.imshow(inverse_rotated_image)
plt.axis('off')

plt.tight_layout()
plt.show()