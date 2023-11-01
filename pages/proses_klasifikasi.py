import tkinter as tk
from tkinter import PhotoImage, filedialog

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

    def create_image_label_button(image_path, label_text, row, column):
        nonlocal segmentation_button, classification_button

        # Gambar
        image = PhotoImage(file=image_path)
        image = image.subsample(2)
        label = tk.Label(image_frame, image=image)
        label.image = image
        label.configure(width=150, height=150)
        label.grid(row=row, column=column, padx=10, pady=10, sticky="nsw")

        # Label untuk teks di bawah gambar
        label_text = tk.Label(image_frame, text=label_text, font=("Arial", 12))
        label_text.grid(row=row + 1, column=column, padx=10, pady=10, sticky="nsew")

        # Tombol
        if label_text.cget("text") == "Gambar Input":
            button_text = "Import Image"
            button = tk.Button(image_frame, text=button_text, command=lambda: import_image(label, label_text))
            button.grid(row=row + 2, column=column, padx=10, pady=10, sticky="nsew")
        elif label_text.cget("text") == "Gambar Segmentasi":
            button_text = "Segmentation Image"
            segmentation_button = tk.Button(image_frame, text=button_text, command=lambda: button_click(label_text), state=tk.DISABLED)
            segmentation_button.grid(row=row + 2, column=column, padx=10, pady=10, sticky="nsew")
        elif label_text.cget("text") == "Gambar Klasifikasi":
            button_text = "Classification Image"
            classification_button = tk.Button(image_frame, text=button_text, command=lambda: button_click(label_text), state=tk.DISABLED)
            classification_button.grid(row=row + 2, column=column, padx=10, pady=10, sticky="nsew")

    def import_image(label, label_text):
        nonlocal segmentation_button, classification_button
        file_path = filedialog.askopenfilename(title="Pilih Gambar", filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
        if file_path:
            # Ganti gambar "Gambar Input" dengan gambar yang dipilih
            image = PhotoImage(file=file_path)
            image = image.subsample(2)
            label.configure(image=image)
            label.image = image
            label_text.configure(text="Gambar Input (Terpilih)")

            # Aktifkan tombol Segmentation dan Classification
            if segmentation_button and classification_button:
                segmentation_button.configure(state=tk.NORMAL)
                classification_button.configure(state=tk.NORMAL)

    # Gambar pertama
    create_image_label_button("images/gambar.png", "Gambar Input", 0, 0)

    # Gambar kedua
    create_image_label_button("images/gambar.png", "Gambar Segmentasi", 0, 1)

    # Gambar ketiga
    create_image_label_button("images/gambar.png", "Gambar Klasifikasi", 0, 2)

    def button_click(label_text):
        print(f"Tombol ditekan untuk: {label_text.cget('text')}")
