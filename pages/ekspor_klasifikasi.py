import tkinter as tk
import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score, multilabel_confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import json
from PIL import Image
import re

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
    report = classification_report(y_test, y_pred, output_dict=True)

    # Menggunakan multilabel_confusion_matrix untuk mendapatkan confusion matrix untuk setiap kelas
    conf_matrix_per_class = multilabel_confusion_matrix(y_test, y_pred)

    # print(f"Accuracy: {accuracy}")
    # print("Classification Report:")
    # print(report)

    # Langkah 8: Optimasi Model
    param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'poly', 'rbf']}
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    # print("Parameter terbaik setelah optimasi:", grid_search.best_params_)

    # Dapatkan parameter optimal w* dan b*
    if best_model.kernel == 'linear':
        best_w = best_model.coef_.ravel()
    else:
        # Jika kernel bukan linear, hitung magnitude vektor bobot
        support_vectors = best_model.support_vectors_
        dual_coefs = best_model.dual_coef_
        best_w = np.dot(dual_coefs, support_vectors)

    # Hitung magnitude vektor bobot
    magnitude_w = np.linalg.norm(best_w)

    # Dapatkan parameter hasil tuning
    best_params_tuned = grid_search.best_params_

    # Function to get true label from file path
    def get_true_label(file_path):
        # Extract the file name from the path
        file_name = os.path.basename(file_path)
        print(f"File name: {file_name}")
        class_names = ["busuk", "kehijauan", "kering", "matang"]
        # Split the file name using underscores
        parts = file_name.split('_')
        print(f"{parts}")

        # Iterate through parts to find the label
        for part in parts:
            print(f"{part}")
            if part.lower() in ["busuk", "kehijauan", "kering", "matang"]:
                # Map the label to the index of class_names
                label_index = class_names.index(part.lower())

                # Print statements for debugging
                print(f"File name: {file_name}")
                print(f"Extracted true label: (Class: {part.lower()})")
                print(f"Index of class_names: {label_index}")

                return label_index

        # If no label is found, return a default label
        return -1  # Replace with the appropriate default label or handle this case as needed


    # Langkah 9: Prediksi pada Data Uji
    new_image_dir = "images/classification/test"
    new_image_paths = [os.path.join(new_image_dir, filename) for filename in os.listdir(new_image_dir) if filename.endswith(".jpg")]

    new_data = []
    dummy_true_labels = []

    for image_path in new_image_paths:
        true_label = get_true_label(image_path)
        
        if true_label != -1:  # Assuming -1 is the default label when no match is found
            dummy_true_labels.append(true_label)
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
    report_test = classification_report(y_test, y_pred_test, output_dict=True)
    # Menggunakan multilabel_confusion_matrix untuk mendapatkan confusion matrix untuk setiap kelas
    conf_matrix_per_class_test = multilabel_confusion_matrix(y_test, y_pred_test)

    # Convert the NumPy array to a nested list
    serializable_results = predictions.tolist()
    
    # Simpan hasil klasifikasi dalam bentuk JSON
    classification_results = {
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix_per_class": [
            {"class": i,
            "confusion_matrix": conf_matrix.tolist(),
            "true_positive": int(conf_matrix[1, 1]),
            "false_positive": int(conf_matrix[0, 1]),
            "false_negative": int(conf_matrix[1, 0]),
            "true_negative": int(conf_matrix[0, 0])}
            for i, conf_matrix in enumerate(conf_matrix_per_class)
        ],
        "best_model_parameters": {
            "C": best_model.C,
            "kernel": best_model.kernel,
            "magnitude_w": magnitude_w,  # Menambahkan magnitude vektor bobot
            "b_star": best_model.intercept_.tolist(),
            "w_star": best_w.tolist()
        },
        "best_tuning_parameters": best_params_tuned,
        "predictions": serializable_results,
        "true_labels": dummy_true_labels,
        "test_accuracy": accuracy_test,
        "test_classification_report": report_test,
        "test_confusion_matrix_per_class": [
            {"class": i,
            "confusion_matrix": conf_matrix.tolist(),
            "true_positive": int(conf_matrix[1, 1]),
            "false_positive": int(conf_matrix[0, 1]),
            "false_negative": int(conf_matrix[1, 0]),
            "true_negative": int(conf_matrix[0, 0])}
            for i, conf_matrix in enumerate(conf_matrix_per_class_test)
        ],
    }

    # Convert NumPy arrays to lists for JSON serialization
    # classification_results["predictions"] = classification_results["predictions"].astype(int).tolist()
    # print(classification_results["predictions"])

    # Save classification results to JSON
    with open("classification_results.json", "w") as json_file:
        json.dump(classification_results, json_file)

    print("Hasil klasifikasi disimpan dalam file classification_results.json")

def show_ekspor_klasifikasi(content_frame):
    label = tk.Label(content_frame, text="Halaman Ekspor Klasifikasi", font=("Arial", 24))
    label.pack()

    # Button untuk melatih model dan menyimpan hasil klasifikasi
    export_button = tk.Button(content_frame, text="Latih dan Ekspor Hasil Klasifikasi", command=train_and_export_classification_results)
    export_button.pack(pady=10)