import tkinter as tk

def show_home(content_frame):
    label = tk.Label(content_frame, text="KLASIFIKASI KUALITAS CABAI RAWIT MERAH DI DAERAH TRENGGALEK MENGGUNAKAN METODE SVM", font=("Arial", 20), wraplength=400, padx= 20, pady= 20)
    label.pack()

    # Informasi tentang pembuat aplikasi
    info_label = tk.Label(content_frame, text="OLEH :\nADE KURNIADI\n19.1.03.02.0063", font=("Arial", 14))
    info_label.pack()

# ...
