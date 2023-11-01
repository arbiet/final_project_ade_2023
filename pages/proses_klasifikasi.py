import tkinter as tk
from tkinter import PhotoImage, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

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

    # Path gambar terpilih
    selected_image_path = None

    def create_image_label_button(image_path, label_text, row, column, width, height):
        nonlocal segmentation_button, classification_button

        # Gambar
        pil_image = Image.open(image_path)
        tk_image = ImageTk.PhotoImage(pil_image)
        image_label = tk.Label(image_frame, image=tk_image)
        image_label.image = tk_image
        image_label.configure(width=width, height=height)
        image_label.grid(row=row, column=column, padx=10, pady=10, sticky="nsw")

        # Label untuk teks di bawah gambar
        label = tk.Label(image_frame, text=label_text, font=("Arial", 12))
        label.grid(row=row + 1, column=column, padx=10, pady=10, sticky="nsew")

        # Tombol
        if label_text == "Gambar Input":
            button_text = "Import Image"
            import_button = tk.Button(image_frame, text=button_text, command=lambda: import_image(image_label, label, segmentation_button, classification_button))
            import_button.grid(row=row + 2, column=column, padx=10, pady=10, sticky="nsew")
        elif label_text == "Gambar Segmentasi":
            button_text = "Segmentation Image"
            segmentation_button = tk.Button(image_frame, text=button_text, command=lambda: segmentation_image(image_label, label, selected_image_path), state=tk.DISABLED)
            segmentation_button.grid(row=row + 2, column=column, padx=10, pady=10, sticky="nsew")
        elif label_text == "Gambar Klasifikasi":
            button_text = "Classification Image"
            classification_button = tk.Button(image_frame, text=button_text, command=lambda: button_click(label), state=tk.DISABLED)
            classification_button.grid(row=row + 2, column=column, padx=10, pady=10, sticky="nsew")

    def import_image(image_label, label, segmentation_button, classification_button):
        nonlocal selected_image_path
        file_path = filedialog.askopenfilename(title="Pilih Gambar", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
        if file_path:
            selected_image_path = file_path  # Simpan path gambar terpilih

            # Buka gambar menggunakan Pillow (PIL)
            pil_image = Image.open(file_path)

            # Dapatkan ukuran gambar
            width, height = pil_image.size

            # Ganti gambar "Gambar Input" dengan gambar yang dipilih
            pil_image = pil_image.resize((width, height))
            tk_image = ImageTk.PhotoImage(pil_image)
            image_label.configure(image=tk_image)
            image_label.image = tk_image
            label.configure(text=f"Gambar Input (Terpilih)\nWidth: {width}, Height: {height}")

            # Aktifkan tombol Segmentation dan Classification
            if segmentation_button and classification_button:
                segmentation_button.configure(state=tk.NORMAL)
                classification_button.configure(state=tk.NORMAL)

    def segmentation_image(image_label, label, image_path):
        # Dapatkan gambar dari label "Gambar Input (Terpilih)"
        image = image_label.cget("image")

        if image:
            # Konversi gambar Tkinter (PIL) ke gambar OpenCV
            tk_image = ImageTk.getimage(image)
            open_cv_image = np.array(tk_image)
            open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

            # Lakukan segmentasi gambar menggunakan Adaptive Thresholding
            gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
            segmented_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

            # Tampilkan gambar hasil segmentasi pada "Gambar Segmentasi"
            pil_segmented_image = Image.fromarray(segmented_image)
            tk_segmented_image = ImageTk.PhotoImage(pil_segmented_image)
            image_label.configure(image=tk_segmented_image)
            image_label.image = tk_segmented_image

    # Gambar pertama
    create_image_label_button("images/gambar.png", "Gambar Input", 0, 0, 150, 150)

    # Gambar kedua
    create_image_label_button("images/gambar.png", "Gambar Segmentasi", 0, 1, 150, 150)

    # Gambar ketiga
    create_image_label_button("images/gambar.png", "Gambar Klasifikasi", 0, 2, 150, 150)

    def button_click(label):
        print(f"Tombol ditekan untuk: {label.cget('text')}")

# Memanggil fungsi show_proses_klasifikasi jika Anda ingin mengeksekusi tampilan ini.
