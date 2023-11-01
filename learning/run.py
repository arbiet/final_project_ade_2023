import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pickle

# Fungsi untuk memuat gambar dari folder
def load_images(folder_path, label, image_size=(128, 128)):
    # Inisialisasi daftar untuk gambar dan label
    images = []
    labels = []
    
    # Iterasi melalui file dalam folder
    for image_name in os.listdir(folder_path):
        # Buat path lengkap ke gambar
        image_path = os.path.join(folder_path, image_name)
        
        # Baca gambar dan konversi ke grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize gambar ke ukuran yang ditentukan
        image = cv2.resize(image, image_size)
        
        # Terapkan ambang adaptif pada gambar
        segmented_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Tambahkan gambar dan label ke daftar
        images.append(segmented_image)
        labels.append(label)
    
    # Kembalikan daftar gambar dan label
    return images, labels

# Memuat gambar dari tiga folder pelatihan
folder_muda = "folder_muda"
folder_kering = "folder_kering"
folder_busuk = "folder_busuk"
folder_matang = "folder_matang"

# Memuat gambar dari folder-folder tersebut dengan label yang sesuai
sehat_images, sehat_labels = load_images(folder_matang, 0)
kering_images, kering_labels = load_images(folder_muda, 1)
busuk_images, busuk_labels = load_images(folder_busuk, 2)

# Menggabungkan semua data gambar dan label
images = sehat_images + kering_images + busuk_images
labels = sehat_labels + kering_labels + busuk_labels

# Membuat array NumPy dari gambar dan label
X = np.array(images).reshape(len(images), -1)
y = np.array(labels)

# Membagi data pelatihan menjadi data pelatihan dan validasi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat dan melatih model SVM
svm_model = svm.SVC(kernel='linear', C=1)
svm_model.fit(X_train, y_train)

# Menguji model pada data validasi
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Cetak akurasi model
print("Accuracy:", accuracy)

# Simpan model SVM jika akurasi lebih dari 70%
model_filename = "svm_model.joblib"  # Nama file model
if accuracy > 0.7:
    script_directory = os.path.dirname(__file__)
    model_path = os.path.join(script_directory, model_filename)
    joblib.dump(svm_model, model_path)
    print(f"Model SVM disimpan sebagai {model_path}")

# Memuat model SVM jika sudah ada
if os.path.isfile(model_filename):
    svm_model = joblib.load(model_filename)

# Memuat gambar dari folder pengujian (folder_test)
folder_test = "folder_test"
test_images, test_labels = load_images(folder_test, 0)  # Menggunakan label 0 untuk pengujian
X_test = np.array(test_images).reshape(len(test_images), -1)

# Melakukan klasifikasi pada gambar pengujian
y_pred_test = svm_model.predict(X_test)
print("Predictions for test images:", y_pred_test)
