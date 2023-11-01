import os
import cv2
import numpy as np
from sklearn import svm
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from run import X, y

# Sekarang Anda dapat menggunakan X dan y di sini

# Fungsi untuk memuat model SVM
def load_svm_model(model_filename):
    if os.path.isfile(model_filename):
        svm_model = svm.SVC(kernel='linear', C=1)
        svm_model.fit(X, y)
        return svm_model
    else:
        return None

# Fungsi untuk memuat gambar dari folder
def load_images(folder_path, label, image_size=(128, 128)):
    images = []
    labels = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, image_size)
        segmented_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        images.append(segmented_image)
        labels.append(label)
    return images, labels

# Membuat dan melatih model SVM
svm_model = svm.SVC(kernel='linear', C=1)
svm_model.fit(X, y)

# Membaca gambar dan label dari folder pelatihan
folder_sehat = "folder_sehat"
folder_kering = "folder_kering"
folder_busuk = "folder_busuk"

sehat_images, sehat_labels = load_images(folder_sehat, 0)
kering_images, kering_labels = load_images(folder_kering, 1)
busuk_images, busuk_labels = load_images(folder_busuk, 2)

# Menggabungkan semua data gambar dan label
images = sehat_images + kering_images + busuk_images
labels = sehat_labels + kering_labels + busuk_labels

# Membuat array NumPy dari gambar dan label
X = np.array(images).reshape(len(images), -1)
y = np.array(labels)

# Fungsi untuk menampilkan gambar pada folder tertentu
def show_images(folder_images, folder_name):
    for i, image in enumerate(folder_images):
        img_label = tk.Label(root, text=f"Image {i + 1}")
        img_label.grid(row=i + 1, column=0)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        image_label = tk.Label(root, image=photo)
        image_label.image = photo
        image_label.grid(row=i + 1, column=1)

# Fungsi untuk menampilkan hasil klasifikasi pada gambar yang diunggah
def classify_image():
    file_path = upload_image()
    if uploaded_image is not None:
        image = uploaded_image
        image = image.reshape(1, -1)
        prediction = svm_model.predict(image)
        class_name = get_class_name(prediction[0])
        result_label.config(text=f"Predicted Class: {class_name}")
        display_image(image)
        if file_path:
            file_label.config(text=f"File Path: {file_path}")

# Fungsi untuk mendapatkan nama kelas berdasarkan angka
def get_class_name(class_id):
    class_names = {
        0: "Daun Cabai Rawit Sehat",
        1: "Daun Cabai Rawit Kering",
        2: "Daun Cabai Rawit Busuk"
    }
    return class_names.get(class_id, "Kelas tidak dikenali")

# Fungsi untuk menampilkan gambar yang diunggah
def display_image(image):
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image=image)
    image_label.config(image=photo)
    image_label.image = photo

# Fungsi untuk mengunggah gambar dari file
def upload_image():
    global uploaded_image
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128))
        segmented_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        uploaded_image = segmented_image
        return file_path
    else:
        return None

# Fungsi utama
if __name__ == "__main__":
    # Membuat antarmuka pengguna
    root = tk.Tk()
    root.title("Image Classification App")

    # Tombol untuk menampilkan gambar dari folder sehat
    show_sehat_button = tk.Button(root, text="Show Sehat Images", command=lambda: show_images(sehat_images, "Sehat"))
    show_sehat_button.grid(row=0, column=0)

    # Tombol untuk menampilkan gambar dari folder kering
    show_kering_button = tk.Button(root, text="Show Kering Images", command=lambda: show_images(kering_images, "Kering"))
    show_kering_button.grid(row=1, column=0)

    # Tombol untuk menampilkan gambar dari folder busuk
    show_busuk_button = tk.Button(root, text="Show Busuk Images", command=lambda: show_images(busuk_images, "Busuk"))
    show_busuk_button.grid(row=2, column=0)

    upload_button = tk.Button(root, text="Upload Image", command=classify_image)
    upload_button.grid(row=3, column=0)

    image_label = tk.Label(root)
    image_label.grid(row=0, column=1, rowspan=4)

    result_label = tk.Label(root, text="Predicted Class: ")
    result_label.grid(row=4, column=0)

    file_label = tk.Label(root, text="File Path: ")
    file_label.grid(row=5, column=0)

    root.mainloop()
