import tkinter as tk

def show_home(content_frame):
    label = tk.Label(content_frame, text="KLASIFIKASI KUALITAS CABAI RAWIT MERAH MENGGUNAKAN METODE SVM", font=("Arial", 20), wraplength=400, padx= 20, pady= 20)
    label.pack()

    # Informasi tentang pembuat aplikasi
    info_label = tk.Label(content_frame, text="OLEH :\nADE KURNIADI\n19.1.03.02.0063", font=("Arial", 14))
    info_label.pack()

    # Label informasi metode SVM
    svm_info_label = tk.Label(content_frame, text="METODE SUPPORT VECTOR MACHINE (SVM):", font=("Arial", 13, "bold"), pady=10)
    svm_info_label.pack()

    svm_description = tk.Label(content_frame, text="Support Vector Machine (SVM) adalah metode pembelajaran mesin yang digunakan untuk klasifikasi dan regresi. SVM mencoba membuat sebuah hyperplane yang memisahkan data ke dalam kelas-kelas tertentu. Dalam konteks klasifikasi citra untuk mengetahui kualitas cabai rawit, SVM dapat digunakan untuk memisahkan citra ke dalam kategori busuk, matang, kehijauan, dan kering.", font=("Arial", 11), wraplength=500, justify=tk.LEFT)
    svm_description.pack()

    # Label informasi penggunaan SVM dalam klasifikasi citra
    classification_info_label = tk.Label(content_frame, text="PENGGUNAAN SVM DALAM KLASIFIKASI CITRA:", font=("Arial", 13, "bold"), pady=10)
    classification_info_label.pack()

    classification_description = tk.Label(content_frame, text="Dalam aplikasi ini, SVM digunakan sebagai algoritma klasifikasi untuk membedakan kualitas cabai rawit merah berdasarkan citra. Data citra yang telah diolah akan dipecah menjadi kategori-kategori seperti busuk, matang, kehijauan, dan kering menggunakan SVM. Algoritma ini dapat membantu dalam proses otomatisasi pengenalan dan klasifikasi kualitas cabai rawit merah.", font=("Arial", 11), wraplength=500, justify=tk.LEFT)
    classification_description.pack()

# ...
