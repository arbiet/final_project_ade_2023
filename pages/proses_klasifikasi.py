import tkinter as tk
from tkinter import PhotoImage

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

    def create_image_label_button(image_path, label_text, row, column):
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
        elif label_text.cget("text") == "Gambar Segmentasi":
            button_text = "Segmentation Image"
        elif label_text.cget("text") == "Gambar Klasifikasi":
            button_text = "Classification Image"

        button = tk.Button(image_frame, text=button_text, command=lambda: button_click(label_text))
        button.grid(row=row + 2, column=column, padx=10, pady=10, sticky="nsew")

    # Gambar pertama
    create_image_label_button("images/gambar.png", "Gambar Input", 0, 0)

    # Gambar kedua
    create_image_label_button("images/gambar.png", "Gambar Segmentasi", 0, 1)

    # Gambar ketiga
    create_image_label_button("images/gambar.png", "Gambar Klasifikasi", 0, 2)

    def button_click(label_text):
        print(f"Tombol ditekan untuk: {label_text.cget('text')}")

    # Anda dapat menyesuaikan path file gambar dan teks label sesuai kebutuhan Anda.
