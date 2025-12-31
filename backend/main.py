from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from app.image_preprocessing import preprocess_image_nlp
from app.ocr_service import detect_text_regions_nlp, extract_text_by_region_nlp
from app.ner_service import extract_fields_nlp
from app.logger import app_logger
import difflib
import easyocr
import re
import cv2
from ultralytics import YOLO

# 1. Initialize Flask properly so Gunicorn can find 'app'
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# 2. Lazy loading or error handling for heavy models
print("Initializing EasyOCR Reader...")
try:
    reader = easyocr.Reader(['en'])
    print("EasyOCR Reader initialized successfully")
except Exception as e:
    print(f"Failed to initialize EasyOCR Reader: {e}")
    reader = None

print("Loading YOLO model...")
try:
    # Ensure this path is correct in your GitHub repo
    yolo_model = YOLO('runs/detect/train2/weights/best.pt')
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    yolo_model = None

@app.route("/")
def server_test():
    return jsonify({"message": "Server is running"})

@app.route('/process-image-nlp', methods=['POST'])
def process_image_nlp():
    if 'image' not in request.files:
        app_logger.error('No image part')
        return jsonify({"error": "No image part"}), 400

    file = request.files['image']
    name_input = request.form.get('name', '')
    university_input = request.form.get('university', '')

    if file.filename == '':
        app_logger.error('No selected file')
        return jsonify({"error": "No selected file"}), 400

    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({"error": "Invalid file type"}), 400

    if reader is None:
        return jsonify({"error": "EasyOCR Reader not initialized"}), 500

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    try:
        _, original_image = preprocess_image_nlp(file_path)
        regions = detect_text_regions_nlp(original_image)
        lines = extract_text_by_region_nlp(regions)
        fields = extract_fields_nlp(lines)
        validation_results = validate_data__nlp(fields, name_input, university_input)
        return jsonify(validation_results), 200
    except Exception as e:
        app_logger.error(f"Processing error: {str(e)}")
        return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

def validate_data__nlp(extracted_fields, name_input, university_input):
    name_extracted = extracted_fields.get('Name') or "Not Recognised"
    university_extracted = extracted_fields.get('University') or "Not Recognised"
    expiration_extracted = extracted_fields.get('Expiration') or "Not Recognised"
    name_similarity = calculate_similarity_nlp(name_input, name_extracted) if name_extracted != "Not Recognised" else "Not Recognised"
    university_similarity = calculate_similarity_nlp(university_input, university_extracted) if university_extracted != "Not Recognised" else "Not Recognised"
    expiration_status = check_expiration_nlp(expiration_extracted)
    return {
        "fields": {"Name": name_extracted, "University": university_extracted, "Expiration": expiration_extracted},
        "name_match": name_similarity,
        "university_match": university_similarity,
        "is_expired": expiration_status,
        "is_valid_card": determine_overall_validity_nlp({"name_match": name_similarity, "university_match": university_similarity, "is_expired": expiration_status})
    }

def calculate_similarity_nlp(input_text, extracted_text):
    input_lower = input_text.lower().strip()
    extracted_lower = extracted_text.lower().strip()
    if input_lower and extracted_lower:
        input_words, extracted_words = set(input_lower.split()), set(extracted_lower.split())
        if input_words.issubset(extracted_words) or extracted_words.issubset(input_words):
            base_similarity = difflib.SequenceMatcher(None, input_lower, extracted_lower).ratio()
            word_match_ratio = len(input_words & extracted_words) / max(len(input_words), len(extracted_words))
            similarity = max(base_similarity, word_match_ratio * 0.9)
        else:
            similarity = difflib.SequenceMatcher(None, input_lower, extracted_lower).ratio()
    else:
        similarity = 0
    return round(similarity * 100, 2)

def check_expiration_nlp(expiration_date_str):
    if expiration_date_str == "Not Recognised": return True
    try:
        return datetime.strptime(expiration_date_str, "%m/%d/%Y") < datetime.now()
    except ValueError: return True

def determine_overall_validity_nlp(validation_results):
    if validation_results['name_match'] == "Not Recognised" or validation_results['university_match'] == "Not Recognised": return False
    if validation_results['is_expired']: return False
    return validation_results['name_match'] >= 60 and validation_results['university_match'] >= 60

# YOLO Functions (Simplified for brevity, keep your original logic here)
def remove_special_characters_yolo(text): return re.sub(r'[^A-Za-z0-9\s/]', '', text)
def correct_ocr_mistakes_yolo(text):
    for inc, corr in {'0': 'O', '1': 'I', '5': 'S'}.items(): text = text.replace(inc, corr)
    return text
def clean_date_format_yolo(text): return re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f'{int(m.group(1)):02}/{int(m.group(2)):02}/{m.group(3)}', text)
def clean_ocr_text_yolo(text): return clean_date_format_yolo(correct_ocr_mistakes_yolo(remove_special_characters_yolo(text)))

def extract_text_from_yolo(image_path, model, save_image_path):
    image = cv2.imread(image_path)
    if image is None: return None
    results = model.predict(source=image_path, save=False)
    detected_text = {"Expiration": [], "Name": [], "University": []}
    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        res = reader.readtext(image[y1:y2, x1:x2], detail=0)
        if res:
            cleaned = [clean_ocr_text_yolo(t) for t in res]
            label = ["Expiration", "Name", "University"][int(cls)]
            detected_text[label].extend(cleaned)
    cv2.imwrite(save_image_path, image)
    return detected_text

@app.route('/process-image-yolo', methods=['POST'])
def process_image_yolo():
    if 'image' not in request.files: return jsonify({"error": "No image part"}), 400
    file = request.files['image']
    if file.filename == '' or reader is None or yolo_model is None: return jsonify({"error": "Init error"}), 500
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + filename)
    try:
        ext_texts = extract_text_from_yolo(file_path, yolo_model, save_path)
        val_res = validate_data_yolo(ext_texts, request.form.get('name', ''), request.form.get('university', ''))
        return jsonify(val_res), 200
    except Exception as e: return jsonify({"error": str(e)}), 500

# Additional Helper Functions (validate_data_yolo, calculate_similarity_yolo, etc. should remain as per your original logic)
# [Your original yolo-related helper functions here...]

# 3. Dynamic Port Binding for Render
if __name__ == "__main__":
    # Render provides a PORT environment variable. If not found, it defaults to 5000.
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

import gc
from flask import after_this_request

@app.after_request
def invoke_gc(response):
    gc.collect()  # Forces Python to free unused memory immediately
    return response
