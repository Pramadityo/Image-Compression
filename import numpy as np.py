import numpy as np
import cv2
import matplotlib.pyplot as plt

def compress_channel(channel, k):
    """Melakukan kompresi pada 1 channel gambar menggunakan SVD"""
    U, S, Vt = np.linalg.svd(channel, full_matrices=False)
    # Simpan hanya k komponen terbesarr
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    Vt_k = Vt[:k, :]
    # Rekonstruksi channel
    compressed_channel = np.dot(U_k, np.dot(S_k, Vt_k))
    return compressed_channel

def compress_image_svd(img, k):
    """Melakukan kompresi gambar berwarna (RGB) menggunakan SVD"""
    compressed_img = np.zeros(img.shape)
    for i in range(3):  # Loop untuk R, G, B
        compressed_img[:, :, i] = compress_channel(img[:, :, i], k)
    # Normalisasi nilai agar valid (0–255)
    compressed_img = np.clip(compressed_img, 0, 255)
    return compressed_img.astype(np.uint8)

def calculate_compression_ratio(original_shape, k):
    """Menghitung rasio kompresi"""
    m, n = original_shape[:2]
    original_size = m * n * 3  # 3 channel
    compressed_size = k * (m + n + 1) * 3
    ratio = compressed_size / original_size
    return ratio

# ==== Program utama ====
# Baca gambar
image_path = 'anjing.png' 
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Parameter k (jumlah singular value yang dipertahankan)
k = 20  # Ganti sesuai kebutuhan (misal: 10, 30, 50, 100)

# Kompresi gambar
compressed_img = compress_image_svd(img, k)

# Hitung rasio kompresi
compression_ratio = calculate_compression_ratio(img.shape, k)
print(f'Compression ratio (approx.): {compression_ratio:.2f}')

# Tampilkan gambar asli dan hasil kompresi
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Gamnbar Asli')
plt.imshow(img)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Hasil Kompresi (k={k})')
plt.imshow(compressed_img)
plt.axis('off')

plt.show()

# Simpan gambar hasil kompresi   
output_path = f'compressed_k{k}.jpg'
cv2.imwrite(output_path, cv2.cvtColor(compressed_img, cv2.COLOR_RGB2BGR))
print(f'Hasil kompresi disimpan sebagai {output_path}')
