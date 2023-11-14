import os

def rename_files(folder_path):
    # Mendapatkan daftar file dalam folder
    files = os.listdir(folder_path)

    # Mengurutkan file berdasarkan waktu pembuatan
    files.sort(key=lambda x: os.path.getctime(os.path.join(folder_path, x)))

    # Mengubah nama file
    for i, file_name in enumerate(files):
        # Mendapatkan ekstensi file (misalnya: .txt, .jpg)
        _, file_extension = os.path.splitext(file_name)

        # Membuat nama baru "gambar_nomorurut"
        new_name = f"test_{i+1}{file_extension}"

        # Menggabungkan path file lama dan baru
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)

        # Mengubah nama file
        try:
            os.rename(old_path, new_path)
            print(f"Berhasil mengubah nama {old_path} menjadi {new_path}")
        except Exception as e:
            print(f"Gagal mengubah nama {old_path}: {str(e)}")

if __name__ == "__main__":
    # Ganti path_folder dengan path folder tempat file-file Anda berada
    path_folder = "folder_test"
    
    # Memanggil fungsi rename_files untuk semua file di dalam folder
    rename_files(path_folder)
