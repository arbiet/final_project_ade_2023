import tkinter as tk
from tkinter import PhotoImage, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from sklearn import svm
import joblib
from sklearn.preprocessing import StandardScaler

def show_proses_klasifikasi(content_frame):
    # Hapus konten saat ini
    for widget in content_frame.winfo_children():
        widget.destroy()

    # Tampilkan konten baru untuk halaman "Proses Klasifikasi"
    label = tk.Label(content_frame, text="Proses Klasifikasi", font=("Arial", 24))
    label.pack()

    # Buat frame untuk gambar-gambar
    image_frame = tk.Frame(content_frame)
    image_frame.pack()

    # Definisi tombol Segmentation dan Classification
    segmentation_button = None
    classification_button = None
    import_button = None

    # Path gambar terpilih
    selected_image_path = None

    # Path untuk gambar hasil segmentasi
    segmented_image_path = None

    # Ukuran gambar yang diinginkan
    image_width = 150
    image_height = 150

    # Folder untuk menyimpan gambar sementara
    temp_folder = "images/temp/"
    os.makedirs(temp_folder, exist_ok=True)  # Buat folder jika belum ada

    def create_image_label_button(image_path, label_text, row, column):
        nonlocal segmentation_button, classification_button, import_button

        # Gambar
        pil_image = Image.open(image_path)
        pil_image = pil_image.resize((image_width, image_height))  # Mengatur ukuran gambar
        tk_image = ImageTk.PhotoImage(pil_image)
        image_label = tk.Label(image_frame, image=tk_image)
        image_label.image = tk_image
        image_label.grid(row=row, column=column, padx=10, pady=10, sticky="nsw")

        # Label untuk teks di bawah gambar
        label = tk.Label(image_frame, text=label_text, font=("Arial", 12))
        label.grid(row=row + 1, column=column, padx=10, pady=10, sticky="nsew")

        # Tombol
        if label_text == "Gambar Input":
            button_text = "Import Image"
            import_button = tk.Button(image_frame, text=button_text, command=lambda: import_image(image_label, label, segmentation_button, import_button))
            import_button.grid(row=row + 2, column=column, padx=10, pady=10, sticky="nsew")
        elif label_text == "Gambar Segmentasi":
            button_text = "Segmentation Image"
            segmentation_button = tk.Button(image_frame, text=button_text, command=lambda: segmentation_image(image_label, label, classification_button), state=tk.DISABLED)
            segmentation_button.grid(row=row + 2, column=column, padx=10, pady=10, sticky="nsew")
        elif label_text == "Gambar Klasifikasi":
            button_text = "Classification Image"
            classification_button = tk.Button(image_frame, text=button_text, command=lambda: classification_image(image_label, label, import_button), state=tk.DISABLED)
            classification_button.grid(row=row + 2, column=column, padx=10, pady=10, sticky="nsew")

    def import_image(image_label, label, segmentation_button, import_button):
        nonlocal selected_image_path, segmented_image_path

        # Hapus gambar hasil segmentasi sementara jika ada
        if segmented_image_path:
            os.remove(segmented_image_path)
            segmented_image_path = None
        file_path = filedialog.askopenfilename(title="Pilih Gambar", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
        if file_path:
            # Simpan gambar terpilih sementara di folder "images/temp/"
            filename = os.path.basename(file_path)
            temp_path = os.path.join(temp_folder, filename)
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            os.replace(file_path, temp_path)
            selected_image_path = temp_path

            # Buka gambar menggunakan Pillow (PIL)
            pil_image = Image.open(temp_path)
            pil_image = pil_image.resize((image_width, image_height))  # Mengatur ukuran gambar
            tk_image = ImageTk.PhotoImage(pil_image)
            image_label.configure(image=tk_image)
            image_label.image = tk_image
            label.configure(text=f"Gambar Input (Terpilih)\nWidth: {image_width}, Height: {image_height}")

            # Aktifkan tombol Segmentation dan Classification
            if segmentation_button:
                import_button.configure(state=tk.DISABLED)
                segmentation_button.configure(state=tk.NORMAL)

    def segmentation_image(image_label, label, classification_button):
        nonlocal segmented_image_path  # Add this line to indicate the use of the global variable

        if selected_image_path:
            # Buka gambar terpilih
            original_image = Image.open(selected_image_path)
            pil_image = Image.open(selected_image_path)

            # Pertimbangkan untuk memproses citra dengan pengaburan atau teknik pemrosesan lainnya
            processed_image = cv2.GaussianBlur(np.array(pil_image), (5, 5), 0)

            # Lakukan segmentasi gambar menggunakan Adaptive Thresholding
            open_cv_image = np.array(processed_image)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

            # Pertimbangkan untuk pemrosesan morfologi (dilasi dan erosi)
            kernel = np.ones((5, 5), np.uint8)
            gray = cv2.dilate(gray, kernel, iterations=1)
            gray = cv2.erode(gray, kernel, iterations=1)

            # Deteksi tepi menggunakan operator Canny
            # Sesuaikan nilai ambang bawah (lower threshold) dan ambang atas (upper threshold)
            lower_threshold = 30  # Sesuaikan sesuai kebutuhan
            upper_threshold = 100  # Sesuaikan sesuai kebutuhan
            edges = cv2.Canny(gray, lower_threshold, upper_threshold)

            # Sebagai contoh, menerapkan Transformasi Hough untuk mendeteksi garis lurus
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=50)

            # Jika ingin menggabungkan garis yang terdeteksi, bisa ditambahkan proses penggabungan di sini
            # Gabungkan garis dengan melakukan dilasi dan operasi penutupan (closing)
            kernel_line = np.ones((5, 5), np.uint8)
            dilated_lines = cv2.dilate(edges, kernel_line, iterations=1)
            closed_lines = cv2.morphologyEx(dilated_lines, cv2.MORPH_CLOSE, kernel_line, iterations=2)

            # Resize the segmented image after line joining
            resized_segmented_image = cv2.resize(closed_lines, (500, 500))

            # Apply noise reduction (Gaussian blur)
            apply_noise_reduction = True  # Set to True if you want to apply noise reduction
            if apply_noise_reduction:
                resized_segmented_image = cv2.GaussianBlur(resized_segmented_image, (3, 3), 0)

            # Invert the segmented image
            inverted_segmented_image = cv2.bitwise_not(resized_segmented_image)

            # Apply the mask to the original image
            original_image_array = np.array(original_image)
            masked_image = cv2.bitwise_and(original_image_array, original_image_array, mask=resized_segmented_image)

            # Save the masked image temporarily
            masked_filename = f"masked_{os.path.basename(selected_image_path)}"
            masked_image_path = os.path.join(temp_folder, masked_filename)
            os.makedirs(os.path.dirname(masked_image_path), exist_ok=True)
            cv2.imwrite(masked_image_path, masked_image)

            # Display the result
            pil_masked_image = Image.fromarray(masked_image)
            pil_masked_image = pil_masked_image.resize((image_width, image_height))
            tk_masked_image = ImageTk.PhotoImage(pil_masked_image)
            image_label.configure(image=tk_masked_image)
            image_label.image = tk_masked_image

            # Set the global variable for the masked image path
            segmented_image_path = masked_image_path  # Overwriting the segmented image path

            # Additional description
            description = f"Gambar Segmentasi\n(Metode: Canny Edge Detection + Hough Transform + Gaussian Blur + Inverted)\n" \
                        f"Masking ke Gambar Asli"
            label.configure(text=description, font=("Arial", 9))

            # Activate the Classification button
            if classification_button:
                segmentation_button.configure(state=tk.DISABLED)
                classification_button.configure(state=tk.NORMAL)

    def classification_image(image_label, label, import_button):
        nonlocal segmented_image_path, selected_image_path

        if segmented_image_path:
            print("Classification function called.")
            # Load the pre-trained SVM model
            model_path = "pages/svm_model.joblib"
            if os.path.exists(model_path):
                model_tuple = joblib.load(model_path)
                model = model_tuple[0]  # Extract the model from the tuple
            else:
                print("Error: SVM model not found.")
                return

            # Load the segmented image
            pil_segmented_image = Image.open(segmented_image_path)
            segmented_image = np.array(pil_segmented_image)

            # Check if the image has three channels (RGB)
            if segmented_image.shape[-1] == 3:
                # Convert to grayscale
                gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
            else:
                # Image is already single-channel
                gray_segmented = segmented_image

            # Resize the segmented image
            resized_segmented_image = cv2.resize(gray_segmented, (500, 500))

            # Flatten the image for classification
            flattened_image = resized_segmented_image.flatten()

            # Scale the features
            scaler = StandardScaler()
            scaled_image = scaler.fit_transform(flattened_image.reshape(1, -1))
            print("Shape of scaled_image:", scaled_image.shape)

            # Make a prediction using the SVM model
            prediction = model.predict(scaled_image)

            # Display the result in the label
            label.configure(text=f"Gambar Klasifikasi\nHasil Klasifikasi: {prediction[0]}")

            # Enable Import Button
            import_button.configure(state=tk.NORMAL)

            # Display the resized and classified image
            pil_classified_image = Image.fromarray(resized_segmented_image)
            pil_classified_image = pil_classified_image.resize((150, 150))
            tk_classified_image = ImageTk.PhotoImage(pil_classified_image)
            image_label.configure(image=tk_classified_image)
            image_label.image = tk_classified_image

    # Gambar pertama
    create_image_label_button("images/gambar.png", "Gambar Input", 0, 0)

    # Gambar kedua
    create_image_label_button("images/gambar.png", "Gambar Segmentasi", 0, 1)

    # Gambar ketiga
    create_image_label_button("images/gambar.png", "Gambar Klasifikasi", 0, 2)

    def button_click(label):
        print(f"Tombol ditekan untuk: {label.cget('text')}")
