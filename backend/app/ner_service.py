import spacy
import re
from app.ocr_service import clean_ocr_text_nlp
nlp = spacy.load("en_core_web_sm")


def extract_fields_nlp(lines):
    name = university = expiration = None
    merged_lines = " ".join(lines)
    merged_lines = clean_ocr_text_nlp(merged_lines)

    doc = nlp(merged_lines)

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text
            break

    for line in lines:
        if re.search(r"University|Institute|College|Polytechnic", line, re.I):
            # Clean the line to extract just the institution name
            # Remove common OCR noise patterns
            cleaned_line = re.sub(r'[^\w\s]', ' ', line)  # Remove special chars except spaces
            # Extract words around the keyword
            words = cleaned_line.split()
            keyword_idx = -1
            for i, word in enumerate(words):
                if re.search(r"University|Institute|College|Polytechnic", word, re.I):
                    keyword_idx = i
                    break
            if keyword_idx >= 0:
                # Take up to 4 words before and after the keyword
                start = max(0, keyword_idx - 3)
                end = min(len(words), keyword_idx + 4)
                university = ' '.join(words[start:end])
            else:
                university = cleaned_line
            break

    for line in lines:
        clean_line = clean_ocr_text_nlp(line)
        # Look for date patterns but exclude enrollment numbers (long numeric strings)
        # Enrollment numbers are usually 10+ digits, dates are shorter
        date_match = re.search(
            r"(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{1,2}/\d{2})", clean_line)
        if date_match:
            date_str = date_match.group(0)
            # Validate it's not an enrollment number (check if it's a valid date format)
            try:
                # Try to parse as date
                from datetime import datetime
                date_found = False
                for fmt in ["%m/%d/%Y", "%m/%d/%y"]:
                    try:
                        datetime.strptime(date_str, fmt)
                        expiration = date_str
                        date_found = True
                        break
                    except ValueError:
                        continue
                if date_found:
                    break
            except:
                continue

    return {"Name": name, "University": university, "Expiration": expiration}
