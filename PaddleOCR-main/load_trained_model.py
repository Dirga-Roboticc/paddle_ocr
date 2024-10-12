from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import os

# Inisialisasi PaddleOCR dengan model terlatih
ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en',
    det_model_dir='./pretrain_models/ch_PP-OCRv4_det_infer',
    # rec_model_dir='./output/db_mv3_bak/best_model',
    # cls_model_dir='./output/db_mv3_bak/best_model'
)

# Fungsi untuk melakukan OCR pada gambar dan menggambar hasilnya
def perform_ocr_and_draw(image_path):
    result = ocr.ocr(image_path, cls=True)
    
    # Buka gambar asli
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # Font untuk teks (ganti path sesuai dengan lokasi font di sistem Anda)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    font = ImageFont.truetype(font_path, 20)

    # Gambar hasil OCR
    for idx, line in enumerate(result):
        if isinstance(line, list) and len(line) == 1:
            line = line[0]  # Ambil elemen pertama jika result adalah list of lists
        if isinstance(line, list) and len(line) >= 2:
            bbox = line[0]
            text = line[1][0] if isinstance(line[1], list) and len(line[1]) > 0 else "Unknown"
            confidence = line[1][1] if isinstance(line[1], list) and len(line[1]) > 1 else 0.0
            
            # Pastikan bbox adalah list of lists dengan 4 titik
            if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(point, list) and len(point) == 2 for point in bbox):
                # Gambar bounding box
                draw.polygon([
                    (int(bbox[0][0]), int(bbox[0][1])),
                    (int(bbox[1][0]), int(bbox[1][1])),
                    (int(bbox[2][0]), int(bbox[2][1])),
                    (int(bbox[3][0]), int(bbox[3][1]))
                ], outline="red", width=2)
                
                # Tulis teks dan confidence
                draw.text((int(bbox[0][0]), int(bbox[0][1]) - 25), f"{text} ({confidence:.2f})", font=font, fill="red")
            else:
                print(f"Warning: Unexpected bbox format for line {idx}: {bbox}")
        else:
            print(f"Warning: Unexpected result format for line {idx}: {line}")

    return img, result

# Contoh penggunaan
if __name__ == "__main__":
    image_path = "../train_data/Screenshot.png"  # Ganti dengan path gambar yang ingin diproses
    output_image, ocr_result = perform_ocr_and_draw(image_path)
    
    # Simpan gambar hasil
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ocr_result.png")
    output_image.save(output_path)
    
    print(f"Hasil OCR telah disimpan di: {output_path}")
    
    # Menampilkan hasil OCR dalam bentuk teks
    for idx, line in enumerate(ocr_result):
        print(f"Line {idx + 1}: {line[1][0]}")
