import tkinter as tk
from tkinter import PhotoImage, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from sklearn import svm
import joblib

# Load the trained SVM model
model_filename = "pages/svm_model.joblib"
svm_model = None
if os.path.isfile(model_filename):
    try:
        svm_model = joblib.load(model_filename)
        print("SVM model loaded successfully.")
    except Exception as e:
        print(f"Error loading SVM model: {e}")
else:
    print(f"SVM model file ('{model_filename}') not found.")

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
        nonlocal selected_image_path
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
        if selected_image_path:
            
            # Buka gambar terpilih
            pil_image = Image.open(selected_image_path)

            # Lakukan segmentasi gambar menggunakan Adaptive Thresholding
            open_cv_image = np.array(pil_image)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

            # Konfigurasi Adaptive Thresholding
            block_size = 11
            constant = 2

            segmented_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, constant)

            # Tampilkan gambar hasil segmentasi pada "Gambar Segmentasi"
            pil_segmented_image = Image.fromarray(segmented_image)
            pil_segmented_image = pil_segmented_image.resize((image_width, image_height))
            tk_segmented_image = ImageTk.PhotoImage(pil_segmented_image)
            image_label.configure(image=tk_segmented_image)
            image_label.image = tk_segmented_image

            # Deskripsi tambahan
            description = f"Gambar Segmentasi\n(Metode: Adaptive Thresholding)"
            label.configure(text=description, font=("Arial", 9))

            # Aktifkan tombol Segmentation dan Classification
            if classification_button:
                segmentation_button.configure(state=tk.DISABLED)
                classification_button.configure(state=tk.NORMAL)

    def classification_image(image_label, label, import_button):
        global svm_model
        print("Classification function called.")
        if selected_image_path and svm_model:
            # Buka gambar terpilih
            pil_image = Image.open(selected_image_path)

            # Resize gambar ke ukuran yang diinginkan
            pil_image_resized = pil_image.resize((128, 128))

            # Konversi gambar ke array NumPy
            open_cv_image = np.array(pil_image_resized)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            flat_image = open_cv_image.flatten()

            try:
                # Resize flat_image to match the expected size used during training
                # Replace 16384 with the actual size expected by your SVM model
                flat_image = flat_image.reshape(-1, 16384)

                # Lakukan klasifikasi menggunakan SVM model
                prediction = svm_model.predict(flat_image)
                class_label = get_class_label(prediction[0])

                # Tampilkan gambar hasil klasifikasi pada "Gambar Klasifikasi"
                tk_classification_image = ImageTk.PhotoImage(pil_image_resized)
                image_label.configure(image=tk_classification_image)
                image_label.image = tk_classification_image

                # Deskripsi tambahan
                description = f"Gambar Klasifikasi\n(Klasifikasi: {class_label})"
                label.configure(text=description, font=("Arial", 9))
                if import_button:
                    import_button.configure(state=tk.NORMAL)
                    segmentation_button.configure(state=tk.NORMAL)
                    classification_button.configure(state=tk.DISABLED)

            except Exception as e:
                print(f"Error during classification: {e}")
        else:
            print("Image path or SVM model is not available.")

    def get_class_label(prediction):
        # Map numeric class label to a human-readable label
        class_labels = {
            0: "Matang",
            1: "Busuk",
            2: "Kehijauan",
            3: "Kering",
        }
        return class_labels.get(prediction, "Unknown")

    # Gambar pertama
    create_image_label_button("images/gambar.png", "Gambar Input", 0, 0)

    # Gambar kedua
    create_image_label_button("images/gambar.png", "Gambar Segmentasi", 0, 1)

    # Gambar ketiga
    create_image_label_button("images/gambar.png", "Gambar Klasifikasi", 0, 2)

    def button_click(label):
        print(f"Tombol ditekan untuk: {label.cget('text')}")
