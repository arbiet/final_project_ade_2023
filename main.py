import tkinter as tk
from pages.home import show_home
from pages.settings import show_settings
from pages.proses_klasifikasi import show_proses_klasifikasi
from pages.hasil_klasifikasi import show_hasil_klasifikasi
from pages.evaluasi_klasifikasi import show_evaluasi_klasifikasi
from pages.ekspor_klasifikasi import show_ekspor_klasifikasi

def change_content(content_frame, new_content):
    # Hapus konten saat ini
    for widget in content_frame.winfo_children():
        widget.destroy()
    # Tampilkan konten baru
    new_content(content_frame)

# Inisialisasi jendela Tkinter
root = tk.Tk()
root.title("Aplikasi Klasifikasi Cabai dengan Metode SVM")
root.iconbitmap("images/favicon.ico")

# Set ukuran jendela menjadi 800x600
root.geometry("800x600")

# Buat frame untuk header dengan nama aplikasi
header_frame = tk.Frame(root, bg="blue")
header_frame.pack(fill="x")
app_name_label = tk.Label(header_frame, text="Aplikasi Klasifikasi Cabai dengan Metode SVM", font=("Arial", 20), fg="white", bg="blue")
app_name_label.pack(side="left")

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
