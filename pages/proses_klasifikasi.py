import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import messagebox
import cv2
import numpy as np
import joblib
from PIL import Image, ImageTk
from sklearn.metrics import accuracy_score
import re
import os
from sklearn.metrics import f1_score
import shutil
import time  # Add this import for timestamp

# Load the trained SVM model
model = joblib.load("model_svm.joblib")

# Load the scaler used during training
scaler = joblib.load("scaler.joblib")

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

def delete_image(panel, edges_panel, label_result, delete_button):
    # Delete the displayed image
    panel.config(image=None)
    panel.image = None
    edges_panel.config(image=None)
    edges_panel.image = None
    # Clear the text in label_result
    label_result.config(text="")

    # Disable the delete button
    delete_button.config(state=tk.DISABLED)

def get_class_name(label):
    class_names = ["busuk", "kehijauan", "kering", "matang"]
    return class_names[label]

def classify_image(image_path, label_result, panel, delete_button, edges_panel):
    # Extract features from the new image
    features = extract_features(image_path)

    # Normalize the features using the same scaler used during training
    features = scaler.transform([features])

    # Predict the class label using the trained model
    prediction = model.predict(features)[0]


    # Display the result in the label
    label_result.config(text=f"Klasifikasi Gambar: {get_class_name(prediction)}")

    # Get the true label from the file path or any other source
    true_label = get_true_label(image_path)

    # Calculate accuracy
    accuracy = 1 if prediction == true_label else 0

    # Calculate F1 score
    f1 = f1_score([true_label], [prediction], average='micro')

    label_result.config(
        text=label_result.cget("text") +
        f"\nAccuracy: {accuracy}\n F1 Score: {f1:.2f}\nPredicted Label: {get_class_name(prediction)}({prediction})\nTrue Label: {get_class_name(true_label)}({true_label})"
    )

    # Display the segmented image
    original_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 50, 150)

    # Convert OpenCV image to PIL format
    edges_pil = Image.fromarray(edges)
    edges_pil.thumbnail((200, 200))
    edges_img = ImageTk.PhotoImage(edges_pil)

    # Update the edges panel with the segmented image
    edges_panel.config(image=edges_img)
    edges_panel.image = edges_img

    # Enable the delete button
    delete_button.config(state=tk.NORMAL)

    # Check if accuracy is above 0.75 before saving the image
    if accuracy > 0.75:
        # Display the segmented image
        original_image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_image, 50, 150)

        # Convert OpenCV image to PIL format
        edges_pil = Image.fromarray(edges)
        edges_pil.thumbnail((200, 200))
        edges_img = ImageTk.PhotoImage(edges_pil)

        # Update the edges panel with the segmented image
        edges_panel.config(image=edges_img)
        edges_panel.image = edges_img

        # Enable the delete button
        delete_button.config(state=tk.NORMAL)

        # Save the image to the corresponding classification folder
        save_image_to_folder(image_path, prediction)
    else:
        # If accuracy is below 0.75, inform the user without saving the image
        label_result.config(
            text=label_result.cget("text") +
            f"\nImage not saved. Accuracy below threshold (0.75)."
        )

    # Retrieve SVM model information
    if hasattr(model, 'support_vectors_') and hasattr(model, 'dual_coef_'):
        # Support Vectors
        support_vectors = model.support_vectors_
        # label_result.config(
        #     text=label_result.cget("text") +
        #     f"\n\nSupport Vectors:\n{support_vectors}"
        # )

        # Hyperplane
        w = model.coef_
        b = model.intercept_
        hyperplane = (features.dot(w.T) + b)[0]
        label_result.config(
            text=label_result.cget("text") +
            f"\n\nHyperplane:\nHyperplane adalah garis pemisah antara dua kelas dalam model ini. \n"
            f"Ia dibentuk oleh kombinasi bobot (w) dan bias (b) dari model SVM. \n"
            f"Jika suatu titik berada di atas Hyperplane, itu mungkin termasuk dalam satu kelas, \n"
            f"dan jika di bawahnya, mungkin termasuk dalam kelas lain. Hyperplane untuk gambar ini adalah: {hyperplane}\n"
        )

        # Decision Scores
        decision_scores = model.decision_function(features)
        label_result.config(
            text=label_result.cget("text") +
            f"\n\nDecision Scores:\nDecision Scores mengukur seberapa dekat suatu sampel data dengan Hyperplane. \n"
            f"Semakin tinggi nilainya, semakin yakin model bahwa sampel data termasuk dalam kelas tertentu. \n"
            f"Sebaliknya, nilai negatif menunjukkan kecenderungan sampel data masuk ke dalam kelas lain. \n"
            f"Untuk gambar ini, Decision Scores-nya adalah: {decision_scores}\n"
        )

        # Menampilkan nilai w dan b
        limited_w = w.flatten()[:5]  # Mengambil lima elemen pertama dari vektor bobot
        label_result.config(
            text=label_result.cget("text") +
            f"\nBobot (w) (lima elemen pertama):\n{limited_w}\n"
            f"Bias (b):\n{b}\n"
        )

    # Display kernel function (if available)
    if hasattr(model, 'kernel'):
        label_result.config(
            text=label_result.cget("text") +
            f"\n\nKernel Function: {model.kernel}"
        )

def save_image_to_folder(image_path, prediction):
    # Extract the file name from the path
    file_name = os.path.basename(image_path)
    
    # Generate a unique identifier based on the current timestamp
    timestamp = int(time.time())
    
    # Construct a new filename with the unique identifier
    new_file_name = f"{file_name.split('.')[0]}_{timestamp}.{file_name.split('.')[-1]}"

    # Define the destination folder based on the predicted label
    destination_folder = f"images/classification/{get_class_name(prediction)}"
    destination_folder_test = f"images/classification/test"

    # Ensure the destination folder exists; create if not
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # Ensure the destination folder exists; create if not
    if not os.path.exists(destination_folder_test):
        os.makedirs(destination_folder_test)

    # Construct the destination path for the image
    destination_path = os.path.join(destination_folder, new_file_name)
    destination_folder_test = os.path.join(destination_folder_test, new_file_name)

    # Copy the image to the destination folder
    try:
        shutil.copy(image_path, destination_path)
        shutil.copy(image_path, destination_folder_test)
        messagebox.showinfo("Image Saved", f"Image saved to: {destination_path}")
    except Exception as e:
        messagebox.showerror("Error", f"Error saving image: {str(e)}")

    print(f"Image saved to: {destination_path}")

def browse_image(label_result, content_frame, panel, delete_button, edges_panel):
    # Open a file dialog to choose an image file
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])

    if file_path:
        # Display the chosen image
        img = Image.open(file_path)
        img.thumbnail((200, 200))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img  # Keep a reference to avoid garbage collection
        panel.pack(pady=10)

        # Enable the delete button
        delete_button.config(state=tk.NORMAL)

        # Classify the image
        classify_image(file_path, label_result, panel, delete_button, edges_panel)

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

def show_proses_klasifikasi(content_frame):
    label = tk.Label(content_frame, text="Halaman Proses Klasifikasi", font=("Arial", 24))
    label.pack(pady=20)
    
    # Label to display the classification result
    label_result = tk.Label(content_frame, text="", font=("Arial", 12))
    label_result.pack(pady=10)

    # Frame to hold the image panels
    image_frame = tk.Frame(content_frame)
    image_frame.pack()

    # Panel to display the original image
    panel = tk.Label(image_frame)
    panel.pack(side=tk.LEFT, padx=10)

    # Panel to display the segmented image
    edges_panel = tk.Label(image_frame)
    edges_panel.pack(side=tk.RIGHT, padx=10)

    # Button to browse for an image
    browse_button = tk.Button(content_frame, text="Pilih Gambar", command=lambda: browse_image(label_result, content_frame, panel, delete_button, edges_panel))
    browse_button.pack()

    # Button to delete the image
    delete_button = tk.Button(content_frame, text="Hapus Gambar", state=tk.DISABLED, command=lambda: delete_image(panel, edges_panel, label_result, delete_button))
    delete_button.pack()
