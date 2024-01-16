import tkinter as tk
from tkinter import ttk
import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image

def extract_features(image_path):
    # Membaca citra menggunakan OpenCV
    image = cv2.imread(image_path)

    # Periksa apakah citra dapat dibaca
    if image is None:
        print(f"Warning: Failed to read image at {image_path}")
        # Berikan nilai khusus atau tanggapi sesuai kebutuhan
        return np.zeros(769)  # Memastikan dimensi fitur tetap

    # Langkah 1: Ekstraksi Ukuran Citra
    size_feature = np.prod(image.shape)

    # Langkah 2: Segmentasi menggunakan deteksi tepi Canny
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)

    # Langkah 3: Ekstraksi Histogram dari hasil segmentasi
    hist_edges = cv2.calcHist([edges], [0], None, [256], [0, 256])

    # Langkah 4: Ekstraksi Histogram Warna dari citra asli
    # Konversi citra ke ruang warna HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Hitung histogram warna untuk setiap saluran (Hue, Saturation, Value)
    hist_hue = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
    hist_saturation = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    hist_value = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])

    # Gabungkan histogram menjadi satu vektor fitur
    color_feature = np.concatenate([hist_hue.flatten(), hist_saturation.flatten(), hist_value.flatten(), hist_edges.flatten()])

    # Perbaikan: Menggunakan np.array daripada np.prod untuk size_feature
    return np.concatenate([np.array([size_feature]), color_feature])

def train_svm_model(progress_var, label_result):
    global file_results  # Gunakan variabel global

    # Langkah 2: Pengumpulan Data Latih
    folder_paths = ["images/classification/busuk", "images/classification/kehijauan", "images/classification/kering", "images/classification/matang"]
    data = []
    labels = []

    # Mendapatkan path absolut dari direktori saat ini
    # current_directory = os.path.dirname(os.path.abspath(__file__))

    for i, folder_name in enumerate(folder_paths):
        # folder_path = os.path.join(current_directory, folder_name)
        
        # Periksa apakah folder ada
        if not os.path.exists(folder_name):
            print(f"Folder {folder_name} tidak ditemukan.")
            continue
        
        file_list = os.listdir(folder_name)
        for file_name in file_list:
            image_path = os.path.join(folder_name, file_name)
            # Langkah 3: Ekstraksi Fitur
            features = extract_features(image_path)
            data.append(features)
            labels.append(i)

    # Menampilkan contoh data latih
    print("Contoh data latih:")
    print("Fitur Citra\t\t\t\t\t\tLabel")
    for i in range(3):
        print(f"{data[i]}\t\t{labels[i]}")

    # Langkah 4: Pembagian Data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Langkah 5: Normalisasi Data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Menampilkan contoh data latih setelah normalisasi
    print("Contoh data latih setelah normalisasi:")
    print("Fitur Citra (Setelah Normalisasi)\t\t\tLabel")
    for i in range(3):
        print(f"{X_train[i]}\t\t{y_train[i]}")

    # Langkah 6: Pelatihan Model SVM
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)

    # Langkah 7: Validasi Model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    # Langkah 8: Optimasi Model
    # Contoh optimasi dengan Grid Search untuk mencari parameter terbaik
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear']}

    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    # Model terbaik setelah optimasi
    best_model = grid_search.best_estimator_

    # Menampilkan parameter terbaik
    print("Parameter terbaik setelah optimasi:", grid_search.best_params_)

    # Langkah 9: Prediksi
    # Misalnya, kita menggunakan beberapa citra uji yang tidak terlibat dalam pelatihan
    new_image_paths = ["images/classification/test/kering_2.jpg", "images/classification/test/matang_6.jpg", "images/classification/test/busuk_19.jpg", "images/classification/test/kehijauan_1.jpg", "images/classification/test/matang_1.jpg", "images/classification/test/matang_3.jpg", "images/classification/test/matang_4.jpg", "images/classification/test/matang_5.jpg"]
    new_data = []
    dummy_true_labels = [3, 4, 1, 2, 4, 4, 4, 4]  # Sesuaikan dengan label sebenarnya citra uji

    for image_path, true_label in zip(new_image_paths, dummy_true_labels):
        # Periksa keberadaan file sebelum mencoba membaca citra
        # image_path = os.path.join(current_directory, image_path)

        # Langkah 3: Ekstraksi Fitur pada citra uji
        new_features = extract_features(image_path)
        new_data.append(new_features)

    # Langkah 5: Normalisasi Data pada citra uji
    new_data = scaler.transform(new_data)

    # Langkah 9: Prediksi kelas citra uji menggunakan model terbaik
    predictions = best_model.predict(new_data)

    # Menampilkan hasil prediksi
    label_result.config(text=label_result.cget("text") + "\nHasil Prediksi untuk Citra Uji:")
    for i, (prediction, true_label) in enumerate(zip(predictions, dummy_true_labels)):
        label_result.config(text=label_result.cget("text") + f"\nCitra {i+1}: Prediksi Kelas {prediction + 1}, Label Sebenarnya {true_label}")

    # Langkah 10: Evaluasi Hasil
    # Evaluasi hasil prediksi menggunakan metrik evaluasi yang sesuai
    y_pred_test = best_model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    report_test = classification_report(y_test, y_pred_test)

    label_result.config(text=label_result.cget("text") + f"\n\nEvaluasi pada Data Uji:")
    label_result.config(text=label_result.cget("text") + f"\nAccuracy: {accuracy_test}")
    label_result.config(text=label_result.cget("text") + "\nClassification Report:")
    label_result.config(text=label_result.cget("text") + f"\n{report_test}")

    # Menyimpan model ke dalam file
    model_filename = "model_svm.joblib"
    joblib.dump(best_model, model_filename)
    label_result.config(text=label_result.cget("text") + f"\n\nModel SVM telah disimpan sebagai {model_filename}")

    # Menyimpan scaler ke dalam file
    scaler_filename = "scaler.joblib"
    joblib.dump(scaler, scaler_filename)
    label_result.config(text=label_result.cget("text") + f"\n\nScaler telah disimpan sebagai {scaler_filename}")

    # Menyimpan model ke dalam file

def show_latihmodel_klasifikasi(content_frame):
    def train_model_and_display_result():
        global file_results  # Gunakan variabel global

        # Function to train the SVM model and display the result
        progress_var.set(0)  # Reset progress bar
        label_result.config(text="Training is in progress...\n")
        content_frame.update_idletasks()  # Update the content frame

        # Panggil fungsi train_svm_model yang baru
        train_svm_model(progress_var, label_result)

    label = tk.Label(content_frame, text="Halaman Latih Model Klasifikasi", font=("Arial", 24))
    label.pack(pady=20)

    button_train = tk.Button(content_frame, text="Latih Model", command=train_model_and_display_result)
    button_train.pack()

    # Progress bar to indicate the training progress
    progress_var = tk.IntVar()
    progress_bar = ttk.Progressbar(content_frame, variable=progress_var, mode='determinate')
    progress_bar.pack(pady=10)

    label_result = tk.Label(content_frame, text="", font=("Arial", 12), wraplength=400, justify="left")
    label_result.pack(pady=10)

# Gunakan variabel file_results yang diubah menjadi global di awal kodingan
file_results = []