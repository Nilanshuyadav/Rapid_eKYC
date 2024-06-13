import pytesseract
from PIL import Image
import re
import base64
from io import BytesIO
import cv2
import numpy as np
import ftfy

# Tesseract configuration
tesseract_cmd = r'C:\Users\Yash Sharma\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

def adhaar_read_data(text):
    name = None
    dob = None
    adh = None
    sex = None

    text = text.replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())
    text = text.lower()

    name_patterns = [
        re.compile(r'name\s*[:;]?\s*([\w\s]+)', re.IGNORECASE),
        re.compile(r'government of india\s*([\w\s]+)\s*(?:dob|date of birth|year of birth|sex|male|female)', re.IGNORECASE),
        re.compile(r'(\b(?:mr|ms|mrs)\.\s*[\w\s]+)', re.IGNORECASE),
        re.compile(r'([\w\s]+)\s*(?:/|\s)?dob\s*[:;]?\s*\d{2}/\d{2}/\d{4}', re.IGNORECASE),
        re.compile(r'([\w\s]+)\s*(?:/|\s)?aeo\s*/\s*(male|female)', re.IGNORECASE)
    ]
    for pattern in name_patterns:
        match = pattern.search(text)
        if match:
            name = match.group(1).strip().title()
            break

    dob_patterns = [
        re.compile(r'\b(?:dob|date of birth|d.o.b)\s*[:;]?\s*(\d{2}/\d{2}/\d{4})\b', re.IGNORECASE),
        re.compile(r'\b(?:year of birth|yob)\s*[:;]?\s*(\d{4})\b', re.IGNORECASE)
    ]
    for pattern in dob_patterns:
        match = pattern.search(text)
        if match:
            dob = match.group(1).strip()
            break

    if 'female' in text:
        sex = "FEMALE"
    elif 'male' in text:
        sex = "MALE"

    aadhar_patterns = [
        re.compile(r'(\d{4}\s\d{4}\s\d{4})'),
        re.compile(r'(\d{4}\s\d{4}\s\d{4}\s\d{4})')
    ]
    for pattern in aadhar_patterns:
        match = pattern.search(text)
        if match:
            adh = match.group(1).strip()
            break

    data = {
        'Name': name,
        'Date of Birth': dob,
        'Adhaar Number': adh,
        'Sex': sex,
        'ID Type': "Adhaar"
    }

    return data

def extractData(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    var = cv2.Laplacian(img, cv2.CV_64F).var()

    if var < 20:
        raise ValueError("Image is Too Blurry")

    text = pytesseract.image_to_string(Image.fromarray(image), lang='eng')
    text = ftfy.fix_text(text)
    text = ftfy.fix_encoding(text)

    data = adhaar_read_data(text)
    return data

# def fetchData(image_data):
#     image = convert_to_byte(image_data)
#     data = extractData(image)
#     return data

def convert_to_byte(data_url):
    header, encoded = data_url.split(",", 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_data))
    image_np = np.array(image)
    return image_np

def compareFace(face1, face2):
    return 1

def addReport(cursor, conn, username, note):
    try:
        cursor.execute("INSERT INTO report (username, note) VALUES (?, ?)", (username, note))
        conn.commit()
    except Exception as e:
        return str(e)
    return "Report submitted successfully"

def contact(cursor, conn, name, email, subject, message):
    try:
        cursor.execute("""
            INSERT INTO Contact (name, email, subject, message)
            VALUES (?, ?, ?, ?)
        """, (name, email, subject, message))
        conn.commit()
    except Exception as e:
        return str(e)
    return "Your message has been sent. Thank you!"


