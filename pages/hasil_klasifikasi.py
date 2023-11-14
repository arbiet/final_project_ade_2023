import tkinter as tk
import os
from PIL import Image, ImageTk

# Define a global variable for current_index
current_index = 0

def show_hasil_klasifikasi(content_frame):
    # Buat frame untuk menampilkan label hasil klasifikasi
    result_label_frame = tk.Frame(content_frame)
    result_label_frame.pack()

    # Buat label hasil klasifikasi
    label = tk.Label(result_label_frame, text="Hasil Klasifikasi", font=("Arial", 24))
    label.pack()  # Menggunakan pack agar berada di tengah

    # Buat frame untuk tombol-tombol folder
    folder_frame = tk.Frame(result_label_frame)
    folder_frame.pack(side="top")  # Mengatur frame di sebelah kiri

    # Path ke direktori images/classification/
    classification_dir = "images/classification/"

    # Dapatkan daftar folder di dalam direktori classification_dir  
    folders = [f for f in os.listdir(classification_dir) if os.path.isdir(os.path.join(classification_dir, f))]

    # Menghitung lebar tombol agar berukuran sama
    max_folder_width = max(len(folder) for folder in folders)
    button_width = max(max_folder_width + 2, 20)  # 2 spasi ekstra, minimal lebar 20

    # Maksimum jumlah tombol per baris
    max_buttons_per_row = 4

    # Buat tombol untuk setiap folder dan atur dalam grid
    for i, folder in enumerate(folders):
        row, col = divmod(i, max_buttons_per_row)
        folder_button = tk.Button(folder_frame, text=folder.capitalize(), width=button_width, command=lambda f=folder: open_classification_folder(f, content_frame, label, image_frame, nav_frame))
        folder_button.grid(row=row, column=col, sticky="ew")

    # Buat frame untuk menampilkan gambar
    image_frame = tk.Frame(result_label_frame)
    image_frame.pack()

    # Buat frame untuk tombol navigasi
    nav_frame = tk.Frame(result_label_frame)
    nav_frame.pack(side="bottom")

def open_classification_folder(folder_name, content_frame, result_label, image_frame, nav_frame):
    # Fungsi ini akan dijalankan saat tombol folder ditekan.
    # Anda dapat menambahkan logika untuk membuka folder atau melakukan tindakan lain di sini.
    global current_index  # Use the global variable
    # Path ke direktori images/classification/
    classification_dir = "images/classification/"

    # Path ke folder yang dipilih
    selected_folder_path = os.path.join(classification_dir, folder_name)

    # Dapatkan daftar gambar yang valid dalam folder
    images = [f for f in os.listdir(selected_folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]

    # Ambil 12 gambar terakhir
    latest_images = images[-12:]

    # Clear previous images and labels
    for widget in image_frame.winfo_children():
        widget.destroy()

    # Clear previous navigation buttons
    for widget in nav_frame.winfo_children():
        widget.destroy()

    # Ubah teks pada label hasil klasifikasi
    result_label.config(text=f"Hasil Klasifikasi {folder_name.capitalize()}")

    # Tampilkan setiap gambar dalam frame atau pesan jika tidak ada data
    if not latest_images:
        no_data_label = tk.Label(image_frame, text="Belum ada data klasifikasi", font=("Arial", 16))
        no_data_label.pack()
    else:
        # Tampilkan setiap gambar dalam frame
        for i, image_file in enumerate(latest_images):
            image_path = os.path.join(selected_folder_path, image_file)
            image = Image.open(image_path)
            image = image.resize((110, 110))  # Atur ukuran gambar
            tk_image = ImageTk.PhotoImage(image)
            img_label = tk.Label(image_frame, image=tk_image, text=image_file, compound=tk.BOTTOM)
            img_label.image = tk_image
            img_label.grid(row=i // 4, column=i % 4, padx=5, pady=5)

        # Jika jumlah gambar lebih dari 12, tampilkan tombol navigasi
        if len(images) > 12:
            prev_button = tk.Button(nav_frame, text="Prev", command=lambda: show_previous_images(selected_folder_path, image_frame, range_label))
            prev_button.pack(side="left")
            next_button = tk.Button(nav_frame, text="Next", command=lambda: show_next_images(selected_folder_path, image_frame, range_label))
            next_button.pack(side="right")

            # Tampilkan label jumlah total gambar
            total_images_label = tk.Label(nav_frame, text=f"Total Images: {len(images)}")
            total_images_label.pack(side="bottom")

            # Inisialisasi range label
            range_label = tk.Label(nav_frame, text="")
            update_range_label(range_label, images, 0)
            range_label.pack(side="bottom")

def update_range_label(label, images, current_index):
    # Hitung rentang gambar yang ditampilkan
    start_index = current_index + 1
    end_index = min(start_index + 11, len(images))
    total_images = len(images)

    # Perbarui teks pada label
    label.config(text=f"{start_index}-{end_index} of {total_images} images")

def show_previous_images(folder_path, image_frame, range_label):
    global current_index  # Use the global variable
    # Tampilkan 8 gambar sebelumnya dari folder
    images = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]
    current_index = current_index - 12

    if current_index < 0:
        current_index = 0

    show_images(images[current_index:current_index + 12], folder_path, image_frame)
    # Update range label
    update_range_label(range_label, images, current_index)

def show_next_images(folder_path, image_frame, range_label):
    global current_index  # Use the global variable
    # Tampilkan 8 gambar selanjutnya dari folder
    images = [f for f in os.listdir(folder_path) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))]
    current_index = current_index + 12

    if current_index >= len(images):
        current_index = len(images) - 12

    show_images(images[current_index:current_index + 12], folder_path, image_frame)
    # Update range label
    update_range_label(range_label, images, current_index)


def show_images(image_list, folder_path, image_frame):
    # Tampilkan gambar dalam frame
    for i, image_file in enumerate(image_list):
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        image = image.resize((110, 110))  # Atur ukuran gambar
        tk_image = ImageTk.PhotoImage(image)
        img_label = tk.Label(image_frame, image=tk_image, text=image_file, compound=tk.BOTTOM)
        img_label.image = tk_image
        img_label.grid(row=i // 4, column=i % 4, padx=5, pady=5)