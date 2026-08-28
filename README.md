# Lung-See

Sistem klasifikasi penyakit paru-paru berbasis **Single-Input Computer Vision** yang mengintegrasikan YOLOv8, MobileNetV2, Masked Grad-CAM, dan Flask.

## Fitur Utama

- **Deteksi Otomatis ROI Paru:** Menggunakan YOLOv8 untuk memotong area paru-paru secara otomatis sehingga inferensi model klasifikasi fokus pada organ target.
- **Klasifikasi Tiga Kelas:** Mengidentifikasi kondisi paru ke dalam kategori `NORMAL`, `PNEUMONIA`, dan `TBC` menggunakan arsitektur MobileNetV2 yang dioptimalkan.
- **Visualisasi Masked Grad-CAM:** Menghasilkan visualisasi Explainable AI (XAI) yang dibatasi hanya pada area parenkim paru-paru guna menghindari aktivasi semu di luar organ.
- **Sistem Pengaman Berlapis (Barrier Handling):**
  - Skor keyakinan `< 50.0%`: Sistem menolak citra secara otomatis untuk mencegah kesalahan diagnosis pada citra non-medis/buruk.
  - Skor keyakinan `50.0% - 69.9%`: Sistem memberikan status `AMBIGU` disertai peringatan verifikasi medis.
  - Skor keyakinan `>= 70.0%`: Sistem menetapkan status `VALID`.
- **Mode Uji Coba Demo:** Menyediakan sampel citra rontgen siap uji langsung dari antarmuka web tanpa harus mengunggah file manual.

## Struktur Direktori Proyek

```text
lung-see/
│
├── app.py                     # Entry point Flask (Routing, API, & Logic Inferensi)
├── wsgi.py                    # Wrapper WSGI untuk server produksi
├── run_demo.py                # Script pembantu eksekusi demo lokal
├── requirements.txt           # Daftar pustaka dan dependensi Python
├── runtime.txt                # Konfigurasi versi runtime Python
├── Procfile                   # Konfigurasi deployment peladen
├── yolov8n.pt                 # Bobot model lokalisasi ROI YOLOv8
│
├── model/
│   ├── PI-V04_model_vision_fase2.h5   # Bobot model klasifikasi MobileNetV2
│   └── model_vision_fase2.h5          # Cadangan bobot model klasifikasi
│
├── static/
│   ├── css/                   # Berkas stylesheet antarmuka
│   ├── js/                    # Berkas interaktivitas JavaScript
│   ├── img/                   # Aset gambar antarmuka dan sampel demo
│   ├── models/                # Aset 3D model anatomi paru
│   ├── uploads/               # Direktori penyimpanan sementara berkas unggahan
│   └── results/               # Direktori penyimpanan hasil visualisasi Grad-CAM
│
└── templates/                 # Berkas template Jinja2 HTML
    ├── base.html              # Template induk
    ├── index.html             # Halaman beranda
    ├── diagnosa.html          # Halaman formulir diagnosa citra
    ├── result.html            # Halaman penyajian hasil analisis
    ├── pneumonia.html         # Halaman edukasi Pneumonia
    ├── tbc.html               # Halaman edukasi Tuberculosis
    ├── covid.html              # Halaman edukasi Covid-19
    └── tentang_kami.html      # Halaman profil pengembang sistem
```

## Prasyarat Perangkat Lunak

Pastikan perangkat berikut telah terpasang:

- Python 3.10.x (disarankan Python 3.10.12)
- Git
- Peramban web modern seperti Google Chrome, Mozilla Firefox, atau Microsoft Edge

## Panduan Instalasi dan Menjalankan Program

Ikuti langkah-langkah berikut secara berurutan untuk menjalankan proyek di lingkungan komputer lokal.

### 1. Kloning Repositori

Buka Terminal / Command Prompt / PowerShell, lalu jalankan:

```bash
git clone https://github.com/fidhera/lung-see.git
cd lung-see
```

### 2. Membuat dan Mengaktifkan Virtual Environment

Membuat lingkungan isolasi dependensi Python.

**Windows (PowerShell):**

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
python -m venv venv
.\venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Memasang Dependensi Pustaka

Pastikan `venv` telah aktif (terdapat tanda `(venv)` di awal baris perintah), kemudian jalankan:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Menjalankan Aplikasi

Eksekusi file aplikasi utama Flask:

```bash
python app.py
```

Setelah server lokal aktif, terminal akan menampilkan informasi berikut:

```text
Memuat Modul Lokalisasi YOLOv8...
Memuat Model Klasifikasi Single-Input v4 (MobileNetV2)...
Seluruh Modul AI PI-V04 Berhasil Dimuat.
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:8080
```

### 5. Mengakses Antarmuka Sistem

Buka peramban dan masuk ke:

```text
http://127.0.0.1:8080
```

atau:

```text
http://localhost:8080
```

## Mekanisme Pengaman dan Ambang Batas Evaluasi

Sistem menerapkan aturan penanganan `confidence score` secara ketat untuk menjaga integritas hasil inferensi.

| Rentang Confidence | Status Validasi | Tindakan Sistem |
|---|---|---|
| `< 50.00%` | **DITOLAK** | Proses analisis dihentikan seketika. Berkas tidak diproses lebih lanjut untuk menghindari salah diagnosis (*false prediction*). |
| `50.00% - 69.99%` | **AMBIGU** | Hasil klasifikasi ditampilkan disertai pesan peringatan evaluasi klinis bahwa aktivasi fitur berada pada batas bawah kritis. |
| `>= 70.00%` | **VALID** | Hasil klasifikasi dan visualisasi Masked Grad-CAM disajikan sebagai luaran inferensi yang terverifikasi kuat oleh pola model. |

## Penafian Medis

> **Penting:** Hasil analisis dan visualisasi yang disajikan oleh sistem Lung-See murni berbasis komputasi matematis model pembelajaran mesin (*Deep Learning*). Sistem ini ditujukan sebagai alat bantu penyaringan awal (*screening tool*) dan bahan penelitian akademis, bukan sebagai pengganti diagnosis mutlak dari dokter spesialis radiologi atau tenaga medis profesional.
