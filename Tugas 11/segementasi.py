import imageio.v3 as imageio
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar input hasil deteksi tepi dengan Sobel
image_path = "sobel.PNG"  
sobel_image = imageio.imread(image_path, pilmode="L")  # Membaca dalam mode grayscale

# Menampilkan gambar asli
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Gambar Asli (Hasil Sobel)")
plt.imshow(sobel_image, cmap="gray")
plt.axis("off")

# Basic Thresholding
threshold_value = 128  

# Membuat citra biner berdasarkan nilai threshold
binary_image = np.where(sobel_image > threshold_value, 255, 0).astype(np.uint8)

# Menampilkan hasil segmentasi citra
plt.subplot(1, 3, 2)
plt.title(f"Hasil Segmentasi (Threshold = {threshold_value})")
plt.imshow(binary_image, cmap="gray")
plt.axis("off")

# Membuat mask untuk segmentasi
mask = (binary_image == 255).astype(np.uint8)
segmented_image = sobel_image * mask

# Menampilkan hasil segmentasi dengan mask
plt.subplot(1, 3, 3)
plt.title("Hasil Segmentasi dengan Mask")
plt.imshow(segmented_image, cmap="gray")
plt.axis("off")

plt.show()

# Menyimpan hasil segmentasi ke file baru
output_path = "segmented_image.PNG"
imageio.imwrite(output_path, binary_image)
print(f"Hasil segmentasi disimpan ke {output_path}")
