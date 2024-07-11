import pytesseract
from PIL import Image
import re
import base64
from io import BytesIO
import cv2
import numpy as np
import ftfy
import pyodbc
import json
import time
import faceRecognisation as fr
from google import generativeai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold, HarmProbability
from google.generativeai.types.answer_types import FinishReason

# Tesseract configuration
# tesseract_cmd = r'C:\Users\Yash Sharma\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

#This is the API Key
generativeai.configure(api_key="AIzaSyD4F-u7Vf2XoZ3m8RH1qmcDchSyHPlCboQ")

gemini_model = generativeai.GenerativeModel("gemini-1.0-pro-vision-latest")
safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

def add_base64_padding(base64_string):
    return base64_string + '=' * (-len(base64_string) % 4)

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_byte = buffered.getvalue()
    img_base64 = base64.b64encode(img_byte).decode('utf-8')
    return img_base64

def generate_content(query_text, files, temperature, max_output_tokens, top_p):
    start_time = time.time()
    contents = [query_text]
    images = [Image.open(BytesIO(base64.b64decode(add_base64_padding(image_file)))) for image_file in files]
    contents.extend(images)

    response = gemini_model.generate_content(
        contents=contents,
        safety_settings=safety_settings,
        generation_config=GenerationConfig(
            candidate_count=1,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            top_p=top_p
        )
    )
    response.resolve()

    request_tokens = gemini_model.count_tokens(contents).total_tokens
    # print("Request tokens:", request_tokens)

    try:
        text_response = response.text
        # print("Text response:", text_response)
        extracted_info = parse_response(text_response)
    except ValueError as e:
        print(f'Error: {e}')
        return None

    finish_reason = response.candidates[0].finish_reason
    error_reason = 'Response was blocked '

    if finish_reason == FinishReason.MAX_TOKENS:
        error_reason += 'due to the maximum number of tokens specified.'
    elif finish_reason == FinishReason.SAFETY:
        error_reason += 'due to safety reasons.'
        flagged_categories = [rating.category for rating in response.candidates[0].safety_ratings if rating.probability != HarmProbability.NEGLIGIBLE]
        error_reason += f' Flagged categories: {flagged_categories}.'
    elif finish_reason == FinishReason.RECITATION:
        error_reason += 'due to recitation.'
    elif finish_reason == FinishReason.OTHER:
        error_reason += 'due to unknown reasons.'

    # print("Finish reason:", finish_reason)
    # print("Error reason:", error_reason)

    return extracted_info

def parse_response(response_text):
    # print("Response Text to be parsed:", response_text)
    cleaned_text = response_text.replace("```", "").strip()
    # print("Cleaned Response Text:", cleaned_text)
    try:
        response_data = json.loads(cleaned_text)
        extracted_info = {
            "name": response_data.get("name"),
            "dob": response_data.get("dob"),
            "gender": response_data.get("gender"),
            "aadhaar_number": response_data.get("aadhar_number")
        }
    except json.JSONDecodeError as e:
        # print(f"Failed to decode JSON response: {e}")
        return None
    return extracted_info

def insert_to_database(info, image_data):
    server = 'INVL0089'
    database = 'eKYC'
    connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes;"

    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    query = """
    INSERT INTO UserUpload (AadhaarNumber, DateOfBirth, Name, Gender)
    VALUES (?, ?, ?, ?)
    """
    cursor.execute(query, (info['aadhaar_number'], info['dob'], info['name'], info['gender']))
    conn.commit()
    cursor.close()
    conn.close()


def compareFace(image_a, image_b):
    image2 = convert_to_byte(image_b)
    image1 = convert_to_byte(image_a)

    result = fr.verifyFace(image1, image2)
    return result


def convert_to_byte(data_url):
    header, encoded = data_url.split(",", 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_data))
    image1 = image.convert('RGB')
    image_np = np.array(image1)
    return image_np


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

# if __name__ == "__main__":
#     # This block is useful for testing purposes when running dbController.py directly
#     pass
