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

print("Initializing EasyOCR Reader...")
try:
    reader = easyocr.Reader(['en'])
    print("EasyOCR Reader initialized successfully")
except Exception as e:
    print(f"Failed to initialize EasyOCR Reader: {e}")
    reader = None

print("Loading YOLO model...")
try:
    yolo_model = YOLO('runs/detect/train2/weights/best.pt')
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    yolo_model = None
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


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

    # Check if file is an allowed image type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({"error": "Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, TIFF)."}), 400

    if reader is None:
        return jsonify({"error": "EasyOCR Reader not initialized"}), 500

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        app_logger.info(f'Image saved to {file_path}')

        try:
            _, original_image = preprocess_image_nlp(file_path)
            regions = detect_text_regions_nlp(original_image)
            lines = extract_text_by_region_nlp(regions)
            fields = extract_fields_nlp(lines)

            # Validate input data against extracted data
            validation_results = validate_data__nlp(
                fields, name_input, university_input)
            app_logger.info("Extraction and validation successful")
            return jsonify(validation_results), 200
        except Exception as e:
            app_logger.error(f"Processing error: {str(e)}")
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


def validate_data__nlp(extracted_fields, name_input, university_input):
    name_extracted = extracted_fields.get('Name') or "Not Recognised"
    university_extracted = extracted_fields.get(
        'University') or "Not Recognised"
    expiration_extracted = extracted_fields.get(
        'Expiration') or "Not Recognised"

    name_similarity = (
        calculate_similarity_nlp(name_input, name_extracted)
        if name_extracted != "Not Recognised" else "Not Recognised"
    )
    university_similarity = (
        calculate_similarity_nlp(university_input, university_extracted)
        if university_extracted != "Not Recognised" else "Not Recognised"
    )
    expiration_status = check_expiration_nlp(expiration_extracted)

    # Determine if the overall card is valid based on these results
    results = {
        "fields": {
            "Name": name_extracted,
            "University": university_extracted,
            "Expiration": expiration_extracted
        },
        "name_match": name_similarity,
        "university_match": university_similarity,
        "is_expired": expiration_status,
        "is_valid_card": determine_overall_validity_nlp({
            "name_match": name_similarity,
            "university_match": university_similarity,
            "is_expired": expiration_status
        })
    }
    return results


def calculate_similarity_nlp(input_text, extracted_text):
    input_lower = input_text.lower().strip()
    extracted_lower = extracted_text.lower().strip()
    
    # For names, check if all words from input are present in extracted (handles order variations)
    if input_lower and extracted_lower:
        input_words = set(input_lower.split())
        extracted_words = set(extracted_lower.split())
        
        # If all input words are in extracted text, give higher similarity
        if input_words.issubset(extracted_words) or extracted_words.issubset(input_words):
            # Calculate base similarity
            base_similarity = difflib.SequenceMatcher(None, input_lower, extracted_lower).ratio()
            # Boost similarity if words match (even if order differs)
            word_match_ratio = len(input_words & extracted_words) / max(len(input_words), len(extracted_words))
            # Combine both metrics
            similarity = max(base_similarity, word_match_ratio * 0.9)
        else:
            similarity = difflib.SequenceMatcher(None, input_lower, extracted_lower).ratio()
    else:
        similarity = difflib.SequenceMatcher(None, input_lower, extracted_lower).ratio()
    
    return round(similarity * 100, 2)


def check_expiration_nlp(expiration_date_str):
    if expiration_date_str == "Not Recognised":
        return True
    try:
        expiration_date = datetime.strptime(expiration_date_str, "%m/%d/%Y")
        return expiration_date < datetime.now()
    except ValueError:
        return True  # If the expiration date is invalid or unrecognized, mark as expired


def determine_overall_validity_nlp(validation_results):
    if validation_results['name_match'] == "Not Recognised" or validation_results['university_match'] == "Not Recognised":
        return False
    
    # Don't fail validation if expiration is not recognized (some cards don't have expiration dates)
    # Only fail if expiration is explicitly marked as expired
    if validation_results['is_expired'] and validation_results.get('fields', {}).get('Expiration') != "Not Recognised":
        return False

    name_threshold = 60  # Lowered from 70 to handle name order variations
    university_threshold = 60  # Lowered from 70 to handle OCR noise

    name_valid = validation_results['name_match'] >= name_threshold
    university_valid = validation_results['university_match'] >= university_threshold
    expiration_valid = validation_results.get('fields', {}).get('Expiration') == "Not Recognised" or not validation_results['is_expired']

    return name_valid and university_valid and expiration_valid

# YOLO


def remove_special_characters_yolo(text):
    return re.sub(r'[^A-Za-z0-9\s/]', '', text)


# Function to correct common OCR mistakes
def correct_ocr_mistakes_yolo(text):
    corrections = {'0': 'O', '1': 'I', '5': 'S'}
    for incorrect, correct in corrections.items():
        text = text.replace(incorrect, correct)
    return text


# Function to clean date formats
def clean_date_format_yolo(text):
    return re.sub(r'(\d{1,2})/(\d{1,2})/(\d{4})', lambda m: f'{int(m.group(1)):02}/{int(m.group(2)):02}/{m.group(3)}', text)


# Function to clean OCR text (combining all steps)
def clean_ocr_text_yolo(text):
    text = remove_special_characters_yolo(text)
    text = correct_ocr_mistakes_yolo(text)
    text = clean_date_format_yolo(text)
    return text


# Function to crop regions detected by YOLO, apply OCR, and clean text
def extract_text_from_yolo(image_path, model, save_image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return None

    # Run YOLO inference to detect text regions
    results = model.predict(source=image_path, save=False)

    # Dictionary to store texts class-wise
    detected_text = {
        "Expiration": [],
        "Name": [],
        "University": []
    }

    # Iterate through each detected object (bounding boxes + class)
    for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)

        # Crop the region from the image based on bounding box
        cropped_image = image[y1:y2, x1:x2]

        # Use EasyOCR to extract text from the cropped region
        result = reader.readtext(cropped_image, detail=0)
        if result:
            # Clean the OCR output
            result = [clean_ocr_text_yolo(text) for text in result]

            # Store cleaned text based on detected class
            if class_id == 0:  # 'Expiration'
                detected_text["Expiration"].extend(result)
            elif class_id == 1:  # 'Name'
                detected_text["Name"].extend(result)
            elif class_id == 2:  # 'University'
                detected_text["University"].extend(result)

    # Save the image with bounding boxes
    cv2.imwrite(save_image_path, image)
    print(f"Image saved with bounding boxes at {save_image_path}")

    return detected_text


@app.route('/process-image-yolo', methods=['POST'])
def process_image_yolo():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400

    file = request.files['image']
    name_input = request.form.get('name', '')
    university_input = request.form.get('university', '')

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Check if file is an allowed image type
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({"error": "Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP, TIFF)."}), 400

    if reader is None:
        return jsonify({"error": "EasyOCR Reader not initialized"}), 500

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Path to save the image with bounding boxes
        save_image_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 'detected_' + filename)

        # Use the pre-loaded YOLO model
        if yolo_model is None:
            return jsonify({"error": "YOLO model not loaded"}), 500
        model = yolo_model

        try:
            # Detect text regions using YOLO, extract text using OCR, and save image
            extracted_texts = extract_text_from_yolo(
                file_path, model, save_image_path)

            if extracted_texts is None:
                return jsonify({"error": "Failed to load image"}), 500

            # Validate input data against extracted data
            validation_results = validate_data_yolo(
                extracted_texts, name_input, university_input)

            return jsonify(validation_results), 200
        except Exception as e:
            app_logger.error(f"Error processing image with YOLO: {str(e)}")
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 500


def validate_data_yolo(extracted_fields, name_input, university_input):
    # Handle null values for extracted fields
    name_extracted = ' '.join(
        extracted_fields.get('Name', [])) or "Not Recognised"
    university_raw = ' '.join(
        extracted_fields.get('University', [])) or "Not Recognised"
    # Clean university text to remove OCR noise
    if university_raw != "Not Recognised":
        university_extracted = clean_university_text(university_raw)
    else:
        university_extracted = "Not Recognised"
    
    expiration_raw = ' '.join(
        extracted_fields.get('Expiration', [])) or "Not Recognised"
    # Filter out enrollment numbers and only keep valid dates
    if expiration_raw != "Not Recognised":
        expiration_extracted = filter_valid_date(expiration_raw)
    else:
        expiration_extracted = "Not Recognised"

    name_similarity = (
        calculate_similarity_yolo(name_input, name_extracted)
        if name_extracted != "Not Recognised" else "Not Recognised"
    )
    university_similarity = (
        calculate_similarity_yolo(university_input, university_extracted)
        if university_extracted != "Not Recognised" else "Not Recognised"
    )
    expiration_status = check_expiration_yolo(expiration_extracted)

    # Determine if the overall card is valid based on these results
    results = {
        "fields": {
            "Name": name_extracted,
            "University": university_extracted,
            "Expiration": expiration_extracted
        },
        "name_match": name_similarity,
        "university_match": university_similarity,
        "is_expired": expiration_status,
        "is_valid_card": determine_overall_validity_yolo({
            "name_match": name_similarity,
            "university_match": university_similarity,
            "is_expired": expiration_status
        })
    }
    return results


def calculate_similarity_yolo(input_text, extracted_text):
    if input_text and extracted_text != "Not Recognised":
        input_lower = input_text.lower().strip()
        extracted_lower = extracted_text.lower().strip()
        
        # Clean extracted text to remove OCR noise (keep only meaningful words)
        # Extract the most relevant part (usually the longest meaningful word sequence)
        extracted_clean = clean_university_text(extracted_lower)
        
        # For names, check if all words from input are present in extracted (handles order variations)
        input_words = set(input_lower.split())
        extracted_words = set(extracted_clean.split())
        
        # If all input words are in extracted text, give higher similarity
        if input_words.issubset(extracted_words) or extracted_words.issubset(input_words):
            # Calculate base similarity
            base_similarity = difflib.SequenceMatcher(None, input_lower, extracted_clean).ratio()
            # Boost similarity if words match (even if order differs)
            word_match_ratio = len(input_words & extracted_words) / max(len(input_words), len(extracted_words))
            # Combine both metrics
            similarity = max(base_similarity, word_match_ratio * 0.9)
        else:
            similarity = difflib.SequenceMatcher(None, input_lower, extracted_clean).ratio()
        
        return round(similarity * 100, 2)
    return "Not Recognised"


def clean_university_text(text):
    """Clean university text by removing OCR noise and extracting the main institution name"""
    # Common university/institution keywords
    keywords = ['university', 'college', 'institute', 'polytechnic', 'academy', 'school']
    
    # Split into words and filter
    words = text.split()
    cleaned_words = []
    
    # Remove very short words (likely OCR noise) unless they're part of a known pattern
    for word in words:
        # Keep words that are longer than 2 chars, or are keywords
        if len(word) > 2 or word.lower() in keywords:
            # Remove words that are mostly numbers or special characters
            if not re.match(r'^[\d\W]+$', word):
                cleaned_words.append(word)
    
    # Try to find the institution name (usually contains keywords or is a proper noun)
    result = ' '.join(cleaned_words)
    
    # If we find keywords, try to extract text around them
    for keyword in keywords:
        if keyword in result.lower():
            # Find the position and extract surrounding words
            idx = result.lower().find(keyword)
            # Extract a reasonable chunk around the keyword
            words_list = result.split()
            keyword_idx = -1
            for i, w in enumerate(words_list):
                if keyword in w.lower():
                    keyword_idx = i
                    break
            if keyword_idx >= 0:
                # Take words before and after the keyword (up to 3 words each side)
                start = max(0, keyword_idx - 2)
                end = min(len(words_list), keyword_idx + 4)
                result = ' '.join(words_list[start:end])
                break
    
    return result


def filter_valid_date(text):
    """Filter text to only return valid date formats, excluding enrollment numbers"""
    # Look for date patterns: MM/DD/YYYY or MM/DD/YY
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
        r'\d{1,2}/\d{1,2}/\d{2}',  # MM/DD/YY
        r'\d{1,2}-\d{1,2}-\d{4}',  # MM-DD-YYYY
        r'\d{4}-\d{1,2}-\d{1,2}',  # YYYY-MM-DD
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            date_str = match.group(0)
            # Validate it's actually a date, not just numbers
            try:
                # Try different date formats
                for fmt in ["%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%Y-%m-%d"]:
                    try:
                        datetime.strptime(date_str, fmt)
                        return date_str
                    except ValueError:
                        continue
            except:
                continue
    
    # If no valid date found, return "Not Recognised"
    return "Not Recognised"


def check_expiration_yolo(expiration_date_str):
    # Mark as expired (True) if expiration is "Not Recognised"
    if expiration_date_str == "Not Recognised":
        return True
    try:
        # Try different date formats
        for fmt in ["%m/%d/%Y", "%m/%d/%y", "%m-%d-%Y", "%Y-%m-%d"]:
            try:
                expiration_date = datetime.strptime(expiration_date_str, fmt)
                return expiration_date < datetime.now()
            except ValueError:
                continue
        return True  # If no format matches, mark as expired
    except:
        return True  # If the expiration date is invalid or unrecognized, mark as expired


def determine_overall_validity_yolo(validation_results):
    # Mark the card as invalid if name or university is "Not Recognised"
    if validation_results['name_match'] == "Not Recognised" or validation_results['university_match'] == "Not Recognised":
        return False
    
    # Don't fail validation if expiration is not recognized (some cards don't have expiration dates)
    # Only fail if expiration is explicitly marked as expired
    if validation_results['is_expired'] and validation_results.get('fields', {}).get('Expiration') != "Not Recognised":
        return False

    # Define thresholds for name and university matching (lowered to handle variations)
    name_threshold = 60  # Lowered from 70 to handle name order variations
    university_threshold = 60  # Lowered from 70 to handle OCR noise

    name_valid = validation_results['name_match'] >= name_threshold
    university_valid = validation_results['university_match'] >= university_threshold
    expiration_valid = validation_results.get('fields', {}).get('Expiration') == "Not Recognised" or not validation_results['is_expired']

    return name_valid and university_valid and expiration_valid


if __name__ == "__main__":
    app.run(debug=True)
