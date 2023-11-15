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
from PIL import Image

# Add a global variable to store file names and their classification results
file_results = []

def load_images(folder_path, label, image_size=(500, 500), apply_noise_reduction=True):
    images = []
    labels = []

    try:
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)

            # Attempt to read the image
            original_image = Image.open(image_path)
            pil_image = Image.open(image_path)

            # Consider processing the image with blurring or other processing techniques
            processed_image = cv2.GaussianBlur(np.array(pil_image), (5, 5), 0)

            # Perform segmentation using Canny Edge Detection
            open_cv_image = np.array(processed_image)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

            # Consider morphological processing (dilation and erosion)
            kernel = np.ones((5, 5), np.uint8)
            gray = cv2.dilate(gray, kernel, iterations=1)
            gray = cv2.erode(gray, kernel, iterations=1)

            # Detect edges using the Canny operator
            lower_threshold = 30
            upper_threshold = 100
            edges = cv2.Canny(gray, lower_threshold, upper_threshold)

            # Example: Apply Hough Transform to detect straight lines
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)

            # If you want to combine the detected lines, you can add a merging process here
            kernel_line = np.ones((5, 5), np.uint8)
            dilated_lines = cv2.dilate(edges, kernel_line, iterations=1)
            closed_lines = cv2.morphologyEx(dilated_lines, cv2.MORPH_CLOSE, kernel_line, iterations=2)

            # Resize the segmented image after line joining
            resized_segmented_image = cv2.resize(closed_lines, image_size)

            # Apply noise reduction (Gaussian blur)
            if apply_noise_reduction:
                resized_segmented_image = cv2.GaussianBlur(resized_segmented_image, (3, 3), 0)

            # Invert the segmented image
            inverted_segmented_image = cv2.bitwise_not(resized_segmented_image)

            # Append the inverted segmented image to the list
            images.append(inverted_segmented_image)
            labels.append(label)

        return images, labels

    except Exception as e:
        print(f"Error loading images from '{folder_path}': {e}")

def train_svm_model(progress_var, label_result):
    global file_results  # Use a global variable to store file names and their results
    # Function to train the SVM model
    # Adjust the paths based on your project structure
    folder_matang = "images/classification/matang"
    folder_busuk = "images/classification/busuk"
    folder_kehijauan = "images/classification/kehijauan"
    folder_kering = "images/classification/kering"
    folder_test = "images/classification/test"

    # Load training images from different folders
    matang_images, matang_labels = load_images(folder_matang, 0, apply_noise_reduction=True)
    busuk_images, busuk_labels = load_images(folder_busuk, 1, apply_noise_reduction=True)
    kehijauan_images, kehijauan_labels = load_images(folder_kehijauan, 2, apply_noise_reduction=True)
    kering_images, kering_labels = load_images(folder_kering, 3, apply_noise_reduction=True)


    # Combine all data for training
    images = np.vstack((matang_images, busuk_images, kehijauan_images, kering_images))
    labels = np.hstack((matang_labels, busuk_labels, kehijauan_labels, kering_labels))

    # Define class names
    class_names = {
        0: 'Matang',
        1: 'Busuk',
        2: 'Kehijauan',
        3: 'Kering'
    }

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.25, random_state=42)

    # Preprocess the data: Flatten images and scale features
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Create and train the SVM model
    model = svm.SVC(kernel='poly', C=1.0, probability=True, gamma="auto")
    model.fit(X_train_scaled, y_train)

    # Save the model along with class names
    joblib.dump((model, scaler, class_names), 'pages/svm_model.joblib')

    # Make predictions on the test set
    y_pred_test = model.predict(X_test_scaled)

    # Calculate accuracy on the test set
    accuracy_test = accuracy_score(y_test, y_pred_test)

    # Load test images
    test_images, test_labels = load_images(folder_test, None, apply_noise_reduction=True)  # None as label, as it's for testing

    # Preprocess test data
    test_images_flat = np.array(test_images).reshape(len(test_images), -1)
    test_images_scaled = scaler.transform(test_images_flat)

    # Make predictions on the test images
    y_pred_test_images = model.predict(test_images_scaled)

    # Display file names and results in the GUI for the test set
    file_results = list(zip(os.listdir(folder_test), y_pred_test_images))
    file_result_message = "\nFile Names and Classification Results for Test Set:\n"
    # for file_name, result in file_results:
        # file_result_message += f"{file_name}: {class_names[result]}\n"

    # Update the label_result text
    label_result.config(text=f"Training complete. Accuracy on test set: {accuracy_test:.2%}\n" + file_result_message)

def show_latihmodel_klasifikasi(content_frame):
    def train_model_and_display_result():
        global file_results  # Use the global variable

        # Function to train the SVM model and display the result
        progress_var.set(0)  # Reset progress bar
        label_result.config(text="Training is in progress...\n")
        content_frame.update_idletasks()  # Update the content frame

        train_svm_model(progress_var, label_result)

        # Display file names and results in the GUI
        file_result_message = "\nFile Names and Classification Results:\n"
        # for file_name, result in file_results:
            # file_result_message += f"{file_name}: ({result})\n"

        # Update the label_result text
        label_result.config(text=label_result.cget("text") + file_result_message)

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

