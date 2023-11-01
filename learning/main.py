import tkinter as tk
from tkinter import PhotoImage

# Fungsi untuk mengganti konten menu
def change_content(content_frame, new_content):
    # Hapus konten saat ini
    for widget in content_frame.winfo_children():
        widget.destroy()
    # Tampilkan konten baru
    new_content(content_frame)

# Fungsi untuk menampilkan halaman beranda
def show_home(content_frame):
    label = tk.Label(content_frame, text="KLASIFIKASI KUALITAS CABAI RAWIT MERAH DI DAERAH TRENGGALEK MENGGUNAKAN METODE SVM", font=("Arial", 20), wraplength=400, padx= 20, pady= 20)
    label.pack()

    # Informasi tentang pembuat aplikasi
    info_label = tk.Label(content_frame, text="OLEH :\nADE KURNIADI\n19.1.03.02.0063", font=("Arial", 14))
    info_label.pack()

# Fungsi untuk menampilkan halaman pengaturan
def show_settings(content_frame):
    label = tk.Label(content_frame, text="Halaman Pengaturan", font=("Arial", 24))
    label.pack()
    
# Fungsi untuk menampilkan halaman proses klasifikasi
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
    create_image_label_button("gambar.png", "Gambar Input", 0, 0)

    # Gambar kedua
    create_image_label_button("gambar.png", "Gambar Segmentasi", 0, 1)

    # Gambar ketiga
    create_image_label_button("gambar.png", "Gambar Klasifikasi", 0, 2)

    def button_click(label_text):
        print(f"Tombol ditekan untuk: {label_text.cget('text')}")

    # Anda dapat menyesuaikan path file gambar dan teks label sesuai kebutuhan Anda.


# Fungsi untuk menampilkan halaman Hasil Klasifikasi
def show_hasil_klasifikasi(content_frame):
    label = tk.Label(content_frame, text="Halaman Hasil Klasifikasi", font=("Arial", 24))
    label.pack()

# Fungsi untuk menampilkan halaman Evaluasi Klasifikasi
def show_evaluasi_klasifikasi(content_frame):
    label = tk.Label(content_frame, text="Halaman Evaluasi Klasifikasi", font=("Arial", 24))
    label.pack()

# Fungsi untuk menampilkan halaman Ekspor Klasifikasi
def show_ekspor_klasifikasi(content_frame):
    label = tk.Label(content_frame, text="Halaman Ekspor Klasifikasi", font=("Arial", 24))
    label.pack()

# Inisialisasi jendela Tkinter
root = tk.Tk()
root.title("Aplikasi Klasifikasi Cabai dengan Metode SVM")
root.iconbitmap("favicon.ico")

# Set ukuran jendela menjadi 800x600
root.geometry("800x600")

# Buat frame untuk header dengan nama aplikasi dan logo
header_frame = tk.Frame(root, bg="blue")
header_frame.pack(fill="x")
app_name_label = tk.Label(header_frame, text="Aplikasi Klasifikasi Cabai dengan Metode SVM", font=("Arial", 20), fg="white", bg="blue")
app_name_label.pack(side="left")
# Anda dapat menambahkan gambar/logo aplikasi di sini dengan menggunakan PhotoImage

# Buat frame untuk sidebar
sidebar = tk.Frame(root, bg="lightgray")
sidebar.pack(side="left", fill="y")

# Buat frame untuk menu content
content_frame = tk.Frame(root)
content_frame.pack(side="right", fill="both", expand=True)

# Buat tombol-tombol di sidebar yang mengisi seluruh lebar sidebar
home_button = tk.Button(sidebar, text="Beranda", command=lambda: change_content(content_frame, show_home), width=20)
proses_klasifikasi_button = tk.Button(sidebar, text="Proses Klasifikasi", command=lambda: change_content(content_frame, show_proses_klasifikasi), width=20)
hasil_klasifikasi_button = tk.Button(sidebar, text="Hasil Klasifikasi", command=lambda: change_content(content_frame, show_hasil_klasifikasi), width=20)
evaluasi_klasifikasi_button = tk.Button(sidebar, text="Evaluasi Klasifikasi", command=lambda: change_content(content_frame, show_evaluasi_klasifikasi), width=20)
ekspor_klasifikasi_button = tk.Button(sidebar, text="Ekspor Klasifikasi", command=lambda: change_content(content_frame, show_ekspor_klasifikasi), width=20)
settings_button = tk.Button(sidebar, text="Pengaturan", command=lambda: change_content(content_frame, show_settings), width=20)

home_button.pack(fill="x")
proses_klasifikasi_button.pack(fill="x")
hasil_klasifikasi_button.pack(fill="x")
evaluasi_klasifikasi_button.pack(fill="x")
ekspor_klasifikasi_button.pack(fill="x")
settings_button.pack(fill="x")

# Tampilkan halaman beranda saat aplikasi pertama kali dijalankan
show_home(content_frame)

root.mainloop()
