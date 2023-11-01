import tkinter as tk
import os

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

    # Buat tombol untuk setiap folder
    for i, folder in enumerate(folders):
        folder_button = tk.Button(folder_frame, text=folder.capitalize(), width=button_width, command=lambda f=folder: open_classification_folder(f))
        folder_button.pack(side="left", fill="x")  # Mengatur tombol di sebelah kiri dengan jarak antar tombol

def open_classification_folder(folder_name):
    # Fungsi ini akan dijalankan saat tombol folder ditekan.
    # Anda dapat menambahkan logika untuk membuka folder atau melakukan tindakan lain di sini.
    print(f"Folder {folder_name} terbuka.")
