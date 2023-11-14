from PIL import Image
import os

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif"}

def is_image_file(file_name):
    _, file_extension = os.path.splitext(file_name)
    return file_extension.lower() in ALLOWED_EXTENSIONS

def resize_image(input_path, output_path, max_size=(500, 500), max_file_size_kb=100):
    try:
        # Buka gambar
        with Image.open(input_path) as img:
            # Resize gambar
            img.thumbnail(max_size)

            # Simpan gambar yang telah diresize
            img.save(output_path, quality=95)  # Sesuaikan quality sesuai kebutuhan

        # Pastikan ukuran file tidak melebihi batas yang diinginkan
        if os.path.getsize(output_path) / 1024 > max_file_size_kb:
            os.remove(output_path)
            raise Exception(f"Ukuran file {output_path} melebihi {max_file_size_kb} KB. File dihapus.")
    except Exception as e:
        print(f"Gagal mengubah ukuran gambar {input_path}: {str(e)}")

def rename_and_resize_files(folder_path):
    # Mendapatkan daftar file dalam folder
    files = os.listdir(folder_path)

    # Mengurutkan file berdasarkan waktu pembuatan
    files.sort(key=lambda x: os.path.getctime(os.path.join(folder_path, x)))

    # Mengubah nama file dan meresize gambar
    for i, file_name in enumerate(files):
        # Pengecekan apakah file adalah gambar
        if is_image_file(file_name):
            # Membuat nama baru "gambar_nomorurut"
            new_name = f"kering_{i + 1}{os.path.splitext(file_name)[1]}"

            # Menggabungkan path file lama dan baru
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_name)

            # Mengubah nama file dan resize gambar
            try:
                # Jika file baru sudah ada, tambahkan angka unik di belakang nama file
                while os.path.exists(new_path):
                    i += 1
                    new_name = f"kering_{i + 1}{os.path.splitext(file_name)[1]}"
                    new_path = os.path.join(folder_path, new_name)

                os.rename(old_path, new_path)
                print(f"Berhasil mengubah nama {old_path} menjadi {new_path}")

                # Resize gambar
                resize_image(new_path, new_path)
            except Exception as e:
                print(f"Gagal mengubah nama {old_path}: {str(e)}")

if __name__ == "__main__":
    # Ganti path_folder dengan path folder tempat file-file Anda berada
    path_folder = "kering"

    # Memanggil fungsi rename_and_resize_files untuk semua file di dalam folder
    rename_and_resize_files(path_folder)
