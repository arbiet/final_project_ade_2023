import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

def browse_image():
    global image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp")])
    if image_path:
        display_images()

def display_images():
    global image_path

    # Load the image using OpenCV for image processing
    cv_image = cv2.imread(image_path, 0)  # Load image in grayscale

    # Create a PIL image from the original image
    original_image = Image.open(image_path)
    original_image = original_image.resize((200, 200))
    original_photo = ImageTk.PhotoImage(original_image)

    image_label.config(image=original_photo)
    image_label.image = original_photo

    label_gambar_asli.config(text="Gambar Asli")
    label_segmentasi.config(text="Segmentasi Citra")

def segment_image():
    global image_path

    # Load the image using OpenCV for image processing
    cv_image = cv2.imread(image_path, 0)  # Load image in grayscale

    # Apply adaptive thresholding
    segmented_image = cv2.adaptiveThreshold(cv_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Create a PIL image from the segmented image
    segmented_image = Image.fromarray(segmented_image)
    segmented_image = segmented_image.resize((200, 200))
    segmented_photo = ImageTk.PhotoImage(segmented_image)

    segmented_label.config(image=segmented_photo)
    segmented_label.image = segmented_photo

    label_gambar_asli.config(text="Gambar Asli")
    label_segmentasi.config(text="Segmentasi Citra")

# Membuat jendela Tkinter
root = tk.Tk()
root.title("Aplikasi Segmentasi Citra")
root.geometry("400x600")  # Mengatur ukuran jendela

# Membuat judul aplikasi di tengah atas
title_label = tk.Label(root, text="Aplikasi Segmentasi Citra", font=("Helvetica", 16))
title_label.grid(row=0, column=0, columnspan=2)

# Membuat frame untuk baris gambar
image_frame = tk.Frame(root)
image_frame.grid(row=1, column=0, columnspan=2)

# Membuat latar belakang gambar default
default_image = Image.new("RGB", (200, 200), "white")
default_photo = ImageTk.PhotoImage(default_image)

image_label = tk.Label(image_frame, image=default_photo)
image_label.image = default_photo
image_label.grid(row=0, column=0)

# Membuat frame untuk tombol
button_frame = tk.Frame(root)
button_frame.grid(row=2, column=0, columnspan=2)

# Membuat tombol unggah gambar
upload_button = tk.Button(button_frame, text="Unggah Gambar", command=browse_image)
upload_button.grid(row=0, column=0)

# Membuat tombol untuk proses segmentasi
segment_button = tk.Button(button_frame, text="Segmentasi", command=segment_image)
segment_button.grid(row=1, column=0)

# Membuat label untuk teks "Gambar Asli" dan "Segmentasi Citra"
label_gambar_asli = tk.Label(image_frame, text="Gambar Asli", font=("Helvetica", 12))
label_gambar_asli.grid(row=1, column=0)

label_segmentasi = tk.Label(image_frame, text="Segmentasi Citra", font=("Helvetica", 12))
label_segmentasi.grid(row=1, column=1)

# Membuat label untuk gambar hasil segmentasi
segmented_label = tk.Label(image_frame)
segmented_label.grid(row=0, column=1)

root.mainloop()
