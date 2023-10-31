
# Panduan Persiapan Awal SKRIPSI ADE

Ini adalah panduan singkat untuk mempersiapkan lingkungan Anda sebelum menjalankan skrip. Pastikan Anda telah mengikuti langkah-langkah ini untuk memastikan semua dependensi dan pengaturan yang diperlukan telah diatur dengan baik.

## Langkah 1: Instalasi Python

Pastikan Python telah diinstal di sistem Anda. Jika belum, Anda dapat mengunduhnya dari [Python Official Website](https://www.python.org/downloads/).

## Langkah 2: Instalasi Visual Studio Code (Opsional)

Jika Anda ingin menggunakan Visual Studio Code sebagai editor kode, Anda dapat mengunduhnya dari [Visual Studio Code Official Website](https://code.visualstudio.com/).

## Langkah 3: Instalasi Git

Jika Anda belum memiliki Git di sistem Anda, Anda dapat mengunduhnya dari [Git Official Website](https://git-scm.com/downloads).

## Langkah 4: Clone Repository

Gunakan perintah berikut untuk mengklon repository ini dari GitHub:

```bash
git clone https://github.com/ikimukti/skripsi-ade.git
```

## Langkah 5: Membuat Environment (Opsional)

Disarankan untuk membuat lingkungan virtual (virtual environment) untuk mengisolasi dependensi proyek ini. Gunakan venv atau virtualenv untuk membuatnya:

```bash
python -m venv myenv
```

Aktifkan lingkungan virtual:
Windows:

```bash
myenv\Scripts\activate
```

```bash
source myenv/bin/activate
```

## Langkah 6: Menginstal Dependensi

Setelah Anda memastikan lingkungan virtual diaktifkan (jika digunakan), gunakan pip untuk menginstal semua dependensi dari file requirements.txt:

```bash
pip install -r requirements.txt
```

# Panduan Menjalankan Skrip

Setelah Anda menyelesaikan langkah-langkah persiapan awal, berikut adalah langkah-langkah untuk menjalankan skrip ini:

## Langkah 1: Aktifkan Environment (Opsional)

Jika Anda telah membuat lingkungan virtual (virtual environment), pastikan Anda mengaktifkannya sebelum menjalankan skrip. Berikut adalah cara mengaktifkan lingkungan virtual:

- Windows:

```bash
myenv\Scripts\activate
```

- macOS dan Linux:

```bash
source myenv/bin/activate
```

## Langkah 2: Menjalankan Skrip
Sekarang Anda siap untuk menjalankan skrip run.py. Gunakan perintah berikut untuk menjalankannya:

```bash
python run.py
```

Skrip ini akan mulai dieksekusi, dan Anda akan melihat output dan hasil eksekusi di konsol.

Pastikan untuk memantau konsol dan mengikuti instruksi atau pesan kesalahan yang mungkin muncul selama proses eksekusi skrip.

# Panduan Update Repository Lokal

Jika Anda ingin memastikan bahwa repository lokal Anda selalu diperbarui dengan perubahan terbaru dari repository GitHub, Anda dapat mengikuti langkah-langkah ini:

## Langkah 1: Pindah ke Direktori Repository

Pastikan Anda berada di direktori yang sesuai dengan repository lokal Anda. Anda dapat menggunakan perintah `cd` untuk berpindah ke direktori yang benar:

```bash
cd path/ke/github/skripsi-ade
```

## Langkah 2: Sinkronisasi dengan Repository GitHub
Gunakan perintah berikut untuk mengambil perubahan terbaru dari repository GitHub dan memperbarui repository lokal Anda:

```bash
git pull
```

Jika Anda bekerja di cabang (branch) lain selain master, gantilah master dengan nama cabang yang Anda gunakan.

## Langkah 3: Mengatasi Konflik (Opsional)

Jika Anda menghadapi konflik selama proses git pull, Anda perlu mengatasi konflik tersebut. Gunakan alat penggabungan yang sesuai untuk menggabungkan perubahan secara manual.

## Langkah 4: Selesaikan Pembaruan

Setelah pembaruan selesai, repository lokal Anda akan diperbarui dengan perubahan terbaru dari GitHub.

