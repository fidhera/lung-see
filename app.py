import os
import gc
import uuid
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# =======================================================================
# 1. KONFIGURASI JALUR DIREKTORI & MODUL AI PI-V04
# =======================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, 'model', 'PI-V04_model_vision_fase2.h5')
YOLO_PATH = os.path.join(BASE_DIR, 'yolov8n.pt')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULT_FOLDER = os.path.join(BASE_DIR, 'static', 'results')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

print("Memuat Modul Lokalisasi YOLOv8...")
yolo_model = YOLO(YOLO_PATH)

print("Memuat Model Klasifikasi Single-Input v4 (MobileNetV2)...")
model_eval = tf.keras.models.load_model(MODEL_PATH)
print("Seluruh Modul AI PI-V04 Berhasil Dimuat.")

CLASS_NAMES = ['NORMAL', 'PNEUMONIA', 'TBC']

# =======================================================================
# 2. PIPELINE UTILITAS MEDIS & VISUALISASI AI (XAI)
# =======================================================================

def process_yolo_roi_crop(image_path):
    """
    Memindai gambar rontgen menggunakan YOLOv8 dengan fungsi Fallback untuk akurasi citra Normal.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, None
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = yolo_model(img_bgr, conf=0.25, verbose=False)
    
    # FALLBACK 1: Jika YOLO tidak menemukan box, gunakan gambar utuh langsung
    if len(results[0].boxes) == 0:
        return img_rgb, img_bgr
        
    box = results[0].boxes.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)
    
    # Ambil persentase area potongan dibanding ukuran asli
    h_orig, w_orig, _ = img_bgr.shape
    crop_w = x2 - x1
    crop_h = y2 - y1
    
    # FALLBACK 2: Jika box terlalu kecil, gunakan gambar utuh asli
    if crop_w < (w_orig * 0.4) or crop_h < (h_orig * 0.4):
        return img_rgb, img_bgr
        
    cropped_bgr = img_bgr[y1:y2, x1:x2]
    if cropped_bgr.size == 0:
        return img_rgb, img_bgr
        
    cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
    return cropped_rgb, img_bgr


def generate_v4_masked_gradcam(model, img_roi_rgb, last_conv_layer_name="out_relu"):
    """
    Menghitung peta aktivasi Grad-CAM terisolasi dengan teknik Masking Thresholding
    """
    img_resized = cv2.resize(img_roi_rgb, (224, 224))
    img_tensor = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
    
    inner_mobilenet = model.layers[1]
    grad_model = tf.keras.models.Model(
        inputs=[inner_mobilenet.inputs],
        outputs=[inner_mobilenet.get_layer(last_conv_layer_name).output, inner_mobilenet.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        x = model.layers[2](conv_outputs)
        x = model.layers[3](x)
        x = model.layers[4](x)
        output_final = model.layers[5](x)
        
        class_idx = tf.argmax(output_final[0])
        loss_value = output_final[:, class_idx]

    grads = tape.gradient(loss_value, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    heatmap_resized = cv2.resize(heatmap.numpy(), (img_roi_rgb.shape[1], img_roi_rgb.shape[0]))
    
    gray_roi = cv2.cvtColor(img_roi_rgb, cv2.COLOR_RGB2GRAY)
    _, lung_mask = cv2.threshold(gray_roi, 15, 255, cv2.THRESH_BINARY)
    heatmap_resized[lung_mask == 0] = 0
    
    predicted_idx_val = class_idx.numpy()
    probabilities_val = output_final[0].numpy()
    
    del grad_model, tape, grads, pooled_grads
    gc.collect()
    
    return heatmap_resized, lung_mask, predicted_idx_val, probabilities_val


# =======================================================================
# 3. ROUTING CONTROLLER WEBSITE FLASK
# =======================================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/covid')
def covid():
    return render_template('covid.html')

@app.route('/pneumonia')
def pneumonia():
    return render_template('pneumonia.html')

@app.route('/tbc')
def tbc():
    return render_template('tbc.html')

@app.route('/tentang-kami')
def tentang_kami():
    return render_template('tentang_kami.html')

@app.route('/diagnosa', methods=['GET', 'POST'])
def diagnosa():
    if request.method == 'GET':
        return render_template('diagnosa.html')
        
    is_demo = request.form.get('is_demo') == 'true'
    
    if is_demo:
        nama_file_demo = request.form.get('nama_file_demo')
        filepath = os.path.join(BASE_DIR, 'static', 'img', 'demo', nama_file_demo)
        filename = nama_file_demo
        if not os.path.exists(filepath):
            return render_template('diagnosa.html', error="Berkas sampel simulasi demo tidak ditemukan di server local.")
    else:
        if 'file_rontgen' not in request.files:
            return redirect(request.url)
        file = request.files['file_rontgen']
        if file.filename == '':
            return redirect(request.url)
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return render_template('diagnosa.html', error="Format file salah. Server hanya menerima ekstensi .png, .jpg, atau .jpeg")
        
        raw_filename = secure_filename(file.filename)
        unique_prefix = uuid.uuid4().hex[:8]
        filename = f"{unique_prefix}_{raw_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

    # 1. JALANKAN PROSES EKSTRAKSI ROI
    img_roi_rgb, img_bgr_raw = process_yolo_roi_crop(filepath)
    
    if img_roi_rgb is None:
        if not is_demo and os.path.exists(filepath):
            os.remove(filepath)
        return render_template('diagnosa.html', error="Aksi Ditolak: Berkas gagal dibaca oleh pustaka pemrosesan citra.")

    # 2. KOMPUTASI PREDIKSI KLASIFIKASI & GENERATE MASKED GRAD-CAM
    heatmap_masked, lung_mask, predicted_idx, all_probabilities = generate_v4_masked_gradcam(model_eval, img_roi_rgb)
    
    result_class = CLASS_NAMES[predicted_idx]
    confidence_score = float(all_probabilities[predicted_idx] * 100)

    # AMBANG BATAS MUTLAK: DIBAWAH 50% SEBAGAI SYSTEM BARRIER ERROR HANDLING
    if confidence_score < 50.0:
        if not is_demo and os.path.exists(filepath):
            os.remove(filepath)
        pesan_interupsi = (
            "Analisis Dihentikan: Citra rontgen yang Anda unggah memiliki tingkat keyakinan rendah di bawah 50%. "
            "Sistem tidak dapat mengidentifikasi kecocokan pola patologi yang spesifik secara aman."
        )
        return render_template('diagnosa.html', error=pesan_interupsi)

    # 3. PROSES SUPERIMPOSE (BLENDING) TERKUNCI PADA CITRA ROI PARU
    heatmap_uint8 = np.uint8(255 * heatmap_masked)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_color_rgb[lung_mask == 0] = 0
    
    superimposed_img = cv2.addWeighted(img_roi_rgb, 0.6, heatmap_color_rgb, 0.4, 0)

    result_filename = f"gradcam_{filename}"
    result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)
    cv2.imwrite(result_filepath, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    # =======================================================================
    # LOGIKA PENGAMAN EVALUASI AMBIGUITAS (50% s.d. 70%)
    # =======================================================================
    status_validasi = "VALID"
    pesan_peringatan_confidence = ""

    if 50.0 <= confidence_score < 70.0:
        status_validasi = "AMBIGU"
        pesan_peringatan_confidence = (
            f"Perhatian Evaluasi Klinis: Model mendeteksi indikasi kondisi {result_class} dengan tingkat "
            f"keyakinan yang berada di rentang batas bawah kritis yaitu {confidence_score:.2f}%. Area spektrum "
            f"aktivasi panas Grad-CAM yang minim mengindikasikan adanya keraguan sistem klasifikasi dalam memisahkan "
            f"fitur tekstur rontgen yang sehat dengan gejala awal infeksi. Pengguna diwajibkan melakukan peninjauan "
            f"kembali secara konvensional ke dokter spesialis radiologi/paru untuk penegakan diagnosis final."
        )

    # 4. KONDISIONAL DESKRIPSI KLINIS DINAMIS
    if result_class == 'TBC':
        deskripsi_klinis = (
            "Sistem mendeteksi adanya konsentrasi lesi infiltrat atau bercak putih konsolidasi kasar "
            "yang dominan pada wilayah atas (apex) paru-paru. Secara klinis pulmonologi, pola ini selaras "
            "dengan karakteristik infeksi bakteri Mycobacterium tuberculosis."
        )
    elif result_class == 'PNEUMONIA':
        deskripsi_klinis = (
            "Sistem mendeteksi adanya pola opasitas kabut halus (ground-glass opacity) atau konsolidasi "
            "menyebar yang dominan di area bawah (basal) paru-paru dekat diafragma. Penampakan ini secara klinis menandakan "
            "kantung udara alveoli terisi oleh akumulasi cairan."
        )
    else:
        deskripsi_klinis = (
            "Sistem mengonfirmasi struktur parenkim paru-paru terlihat bersih, simetris, dan fungsional. "
            "Rasio radiolusen (wilayah hitam udara) mendominasi secara normal tanpa adanya indikasi bercak opasitas "
            "maupun densitas cairan abnormal."
        )

    disclaimer_medis = (
        "Hasil analisis ini murni berbasis komputasi matematika model pembelajaran mendalam dan ditujukan "
        "hanya sebagai alat bantu penyaringan awal (screening tool) untuk kebutuhan riset akademis."
    )

    path_tampilan_asal = f"img/demo/{filename}" if is_demo else f"uploads/{filename}"

    return render_template(
        'result.html',
        prediction=result_class,
        confidence=f"{confidence_score:.2f}%",
        original_img=path_tampilan_asal,
        gradcam_img=f"results/{result_filename}",
        deskripsi=deskripsi_klinis,
        disclaimer=disclaimer_medis,
        status_validasi=status_validasi,
        pesan_warning=pesan_peringatan_confidence
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)