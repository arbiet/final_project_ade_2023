import tkinter as tk
from tkinter import ttk
import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for feature scaling
import joblib

def load_images(folder_path, label, image_size=(128, 128)):
    images = []
    labels = []

    try:
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            # Attempt to read the image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                # Resize the image
                image = cv2.resize(image, image_size)

                # Apply adaptive threshold
                segmented_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

                images.append(segmented_image)
                labels.append(label)
            else:
                print(f"Warning: Unable to read image '{image_path}'.")

    except Exception as e:
        print(f"Error loading images from '{folder_path}': {e}")

    return images, labels

def train_svm_model(progress_var, label_result):
    # Function to train the SVM model
    # Adjust the paths based on your project structure
    folder_matang = "images/classification/matang"
    # folder_bersemu = "images/classification/bersemu"
    # folder_berjamur = "images/classification/berjamur"
    folder_busuk = "images/classification/busuk"
    folder_kehijauan = "images/classification/kehijauan"
    # folder_luka = "images/classification/luka"
    folder_kering = "images/classification/kering"

    # Load images from different folders
    matang_images, matang_labels = load_images(folder_matang, 0)
    busuk_images, busuk_labels = load_images(folder_busuk, 1)
    kehijauan_images, kehijauan_labels = load_images(folder_kehijauan, 2)
    kering_images, kering_labels = load_images(folder_kering, 3)
    # berjamur_images, berjamur_labels = load_images(folder_berjamur, 2)
    # bersemu_images, bersemu_labels = load_images(folder_bersemu, 1)
    # luka_images, luka_labels = load_images(folder_luka, 5)

    # Combine all data
    # images = matang_images + bersemu_images + berjamur_images + busuk_images + kehijauan_images + luka_images + kering_images
    # labels = matang_labels + bersemu_labels + berjamur_labels + busuk_labels + kehijauan_labels + luka_labels + kering_images

    # Combine all data
    images = matang_images + busuk_images + kehijauan_images + kering_images
    labels = matang_labels + busuk_labels + kehijauan_labels + kering_labels

    if not images:
        label_result.config(text="No images loaded. Please check the image paths.")
        return

    # Convert to NumPy arrays
    X = np.array(images).reshape(len(images), -1)
    y = np.array(labels)

    # Convert to NumPy arrays
    X = np.array(images).reshape(len(images), -1)
    y = np.array(labels)

    # Scale the input features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Create and train the SVM model
    svm_model = svm.SVC(kernel='linear', C=1)
    svm_model.fit(X_train, y_train)

    # Test the model on validation data
    y_pred = svm_model.predict(X_test)

    # Create a list to store accuracy for each file
    file_accuracies = []

    for i, (true_label, predicted_label) in enumerate(zip(y_test, y_pred)):
        file_accuracy = 1 if true_label == predicted_label else 0
        file_accuracies.append(file_accuracy)
        print(f"File {i+1}: Accuracy: {file_accuracy}")

    # Calculate overall accuracy
    overall_accuracy = sum(file_accuracies) / len(file_accuracies)

    # Print model accuracy
    accuracy_message = f"Overall Accuracy: {overall_accuracy}"
    print(accuracy_message)

    # Save the model if overall accuracy is above a certain threshold
    model_filename = "svm_model.joblib"
    script_directory = os.path.dirname(__file__)
    model_path = os.path.join(script_directory, model_filename)

    # Delete existing model file if it exists
    if os.path.exists(model_path):
        os.remove(model_path)
        print(f"Existing model file '{model_path}' deleted.")

    if overall_accuracy > 0.7:
        joblib.dump(svm_model, model_path)
        accuracy_message += f"\nModel SVM saved as {model_path}"
    else:
        accuracy_message += "\nModel SVM not saved due to overall accuracy less than 70%"

    # Update the label_result text
    label_result.config(text=accuracy_message)

def show_latihmodel_klasifikasi(content_frame):
    def train_model_and_display_result():
        # Function to train the SVM model and display the result
        progress_var.set(0)  # Reset progress bar
        label_result.config(text="Training is in progress...\n")
        content_frame.update_idletasks()  # Update the content frame

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
