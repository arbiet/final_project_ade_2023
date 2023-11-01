import tkinter as tk
import os
from PIL import Image, ImageTk

def show_hasil_klasifikasi(content_frame):
    label = tk.Label(content_frame, text="Hasil Klasifikasi", font=("Arial", 24))
    label.pack()  # Menggunakan pack agar berada di tengah

    # Path ke direktori images/classification/
    classification_dir = "images/classification/"

    # Dapatkan daftar folder di dalam direktori classification_dir  
    folders = [f for f in os.listdir(classification_dir) if os.path.isdir(os.path.join(classification_dir, f))]

    # Hitung jumlah folder
    num_folders = len(folders)

    # Buat frame untuk tombol-tombol folder
    folder_frame = tk.Frame(content_frame)
    folder_frame.pack(side="top")  # Mengatur frame di sebelah kiri

    # Menghitung lebar tombol agar berukuran sama
    max_folder_width = max(len(folder) for folder in folders)
    button_width = max(max_folder_width + 2, 20)  # 2 spasi ekstra, minimal lebar 20

    # Maksimum jumlah tombol per baris
    max_buttons_per_row = 4

    # Buat tombol untuk setiap folder dan atur dalam grid
    for i, folder in enumerate(folders):
        row, col = divmod(i, max_buttons_per_row)
        folder_button = tk.Button(folder_frame, text=folder.capitalize(), width=button_width, command=lambda f=folder: open_classification_folder(f))
        folder_button.grid(row=row, column=col, sticky="ew")

    # Buat frame untuk gambar dan atur dalam grid
    image_frame = tk.Frame(content_frame)
    image_frame.pack()

    # Ambil gambar dari setiap folder dan tampilkan
    row_count = 4
    col_count = 6

    for i, folder in enumerate(folders):
        image_dir = os.path.join(classification_dir, folder)
        images = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]

        for j, image_file in enumerate(images):
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path)
            image = image.resize((150, 150))  # Atur ukuran gambar
            tk_image = ImageTk.PhotoImage(image)
            img_label = tk.Label(image_frame, image=tk_image)
            img_label.image = tk_image
            img_label.grid(row=i, column=j, padx=5, pady=5)

def open_classification_folder(folder_name):
    # Fungsi ini akan dijalankan saat tombol folder ditekan.
    # Anda dapat menambahkan logika untuk membuka folder atau melakukan tindakan lain di sini.
    print(f"Folder {folder_name} terbuka.")