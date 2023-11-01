import os
from PIL import Image
import pyheif

# Ganti path_folder_input dengan direktori tempat file HEIC berada
path_folder_input = "convert/"

# Ganti path_folder_output dengan direktori tempat file PNG akan disimpan
path_folder_output = "output/"

# Pastikan direktori output sudah ada atau buat jika belum ada
if not os.path.exists(path_folder_output):
    os.makedirs(path_folder_output)

# Loop melalui semua file di direktori input
for filename in os.listdir(path_folder_input):
    if filename.endswith(".heic"):
        input_file = os.path.join(path_folder_input, filename)
        output_file = os.path.join(path_folder_output, os.path.splitext(filename)[0] + ".png")

        print(f"Mengonversi {input_file} ke {output_file}...")

        heif_file = pyheif.read(input_file)
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )

        # Simpan gambar dalam format PNG
        image.save(output_file, "PNG")

        print(f"Selesai mengonversi {input_file}.")

print("Semua file HEIC telah diubah menjadi PNG.")
