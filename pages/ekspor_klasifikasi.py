import tkinter as tk
import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
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

def train_and_export_classification_results():
    # Langkah 2: Pengumpulan Data Latih
    folder_paths = ["images/classification/busuk", "images/classification/kehijauan", "images/classification/kering", "images/classification/matang"]
    data = []
    labels = []

    for i, folder_name in enumerate(folder_paths):
        if not os.path.exists(folder_name):
            print(f"Folder {folder_name} tidak ditemukan.")
            continue
        
        file_list = os.listdir(folder_name)
        for file_name in file_list:
            image_path = os.path.join(folder_name, file_name)
            features = extract_features(image_path)
            data.append(features)
            labels.append(i)

    # Langkah 4: Pembagian Data
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Langkah 5: Normalisasi Data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf']}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print("Parameter terbaik setelah optimasi:", grid_search.best_params_)

    # Langkah 9: Prediksi pada Data Uji
    new_image_paths = ["images/classification/test/kering_2.jpg", "images/classification/test/matang_6.jpg", "images/classification/test/busuk_19.jpg", "images/classification/test/kehijauan_1.jpg", "images/classification/test/matang_1.jpg", "images/classification/test/matang_3.jpg", "images/classification/test/matang_4.jpg", "images/classification/test/matang_5.jpg"]
    new_data = []
    dummy_true_labels = [3, 4, 1, 2, 4, 4, 4, 4]  # Sesuaikan dengan label sebenarnya citra uji

    for image_path, true_label in zip(new_image_paths, dummy_true_labels):
        new_features = extract_features(image_path)
        new_data.append(new_features)

    # Langkah 5: Normalisasi Data pada citra uji
    new_data = scaler.transform(new_data)

    # Langkah 9: Prediksi kelas citra uji menggunakan model terbaik
    predictions = best_model.predict(new_data)

    # Konversi hasil prediksi menjadi tipe data int
    predictions = predictions.astype(int)

    # Evaluasi hasil prediksi menggunakan metrik evaluasi yang sesuai
    y_pred_test = best_model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    report_test = classification_report(y_test, y_pred_test)

    # Simpan hasil klasifikasi dalam bentuk JSON
    classification_results = {
        "accuracy": accuracy,
        "classification_report": report,
        "best_model_parameters": grid_search.best_params_,
        "predictions": list(predictions),
        "true_labels": dummy_true_labels,
        "test_accuracy": accuracy_test,
        "test_classification_report": report_test
    }

    # Convert NumPy arrays to lists for JSON serialization
    print(classification_results["predictions"])

    # Save classification results to JSON
    # with open("classification_results.json", "w") as json_file:
        # json.dump(classification_results, json_file)

    print("Hasil klasifikasi disimpan dalam file classification_results.json")

def show_ekspor_klasifikasi(content_frame):
    label = tk.Label(content_frame, text="Halaman Ekspor Klasifikasi", font=("Arial", 24))
    label.pack()

    # Button untuk melatih model dan menyimpan hasil klasifikasi
    export_button = tk.Button(content_frame, text="Latih dan Ekspor Hasil Klasifikasi", command=train_and_export_classification_results)
    export_button.pack(pady=10)
