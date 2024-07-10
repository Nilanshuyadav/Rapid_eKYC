from flask import Flask, request, render_template, jsonify
import dbController as db
import base64
import pyodbc
import numpy as np
import cv2
from flask_cors import CORS
from PIL import Image
import io
import time
from google import generativeai
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold, HarmProbability
from google.generativeai.types.answer_types import FinishReason

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Function to connect to the SQL Server database using Windows authentication
def get_db_connection():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=INVL0077;'
        'DATABASE=RapidKyc;'
        'Trusted_Connection=yes;'
    )
    return conn

# Configure GenerativeAI
generativeai.configure(api_key="AIzaSyD4F-u7Vf2XoZ3m8RH1qmcDchSyHPlCboQ")
gemini_model = generativeai.GenerativeModel("gemini-1.0-pro-vision-latest")

safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
}

@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/login", methods=['GET'])
def login():
    return render_template('login.html')

@app.route("/box", methods=['GET'])
def box():
    return render_template('box.html')

@app.route("/step1", methods=['GET'])
def step1():
    return render_template('step1.html')

@app.route("/step2", methods=['GET'])
def step2():
    return render_template('step2.html')

@app.route("/sample", methods=['GET'])
def sample():
    return render_template('sample.html')

@app.route("/step4", methods=['GET'])
def step4():
    return render_template('step4.html')

@app.route("/final", methods=['GET'])
def final():
    return render_template('final.html')

@app.route("/signup", methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        try:
            fullname = request.form['fullname']
            email = request.form['email']
            password = request.form['password']
            repeat_password = request.form['repeatpass']

            # Basic validation
            if password != repeat_password:
                return "Passwords do not match!", 400

            # Insert into database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO Users (FullName, Email, Password)
                VALUES (?, ?, ?)
            """, (fullname, email, password))
            conn.commit()
            cursor.close()
            conn.close()

            return render_template('index.html')  # Render index.html after successful insertion

        except KeyError as e:
            return f"Missing form field: {str(e)}", 400

    return render_template('login.html')

@app.route('/google_login', methods=['POST'])
def google_login():
    # Implement Google login logic here
    # This could involve redirecting to Google's authentication page and handling the callback
    return jsonify({"message": "Google login functionality is under development"}), 200

@app.route('/facebook_login', methods=['POST'])
def facebook_login():
    # Implement Facebook login logic here
    # This could involve redirecting to Facebook's authentication page and handling the callback
    return jsonify({"message": "Facebook login functionality is under development"}), 200


@app.route("/report/addreport", methods=['POST'])
def add_report():
    try:
        username = request.form['username']
        note = request.form['note']

        conn = get_db_connection()
        cursor = conn.cursor()
        result = db.addReport(cursor, conn, username, note)
        cursor.close()
        conn.close()

        return jsonify({"message": result}), 200
    except Exception as e:
        print(f"Error submitting report: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/contact', methods=['POST'])
def contact():
    try:
        name = request.form['name']
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']

        conn = get_db_connection()
        cursor = conn.cursor()
        result = db.contact(cursor, conn, name, email, subject, message)
        cursor.close()
        conn.close()

        return jsonify({"message": result}), 200
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred while submitting your message."}), 500

@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        captured_image = request.form['capturedImage']
        image_data = base64.b64decode(captured_image.split(",")[1])
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        # Convert OpenCV image (numpy array) to PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        query_text = "You are a document verification assistant. Here is a document of a user. Organize all details present in the image to an object."
        files = [db.image_to_base64(pil_image)]  # Convert PIL Image to base64
        temperature = 0.5
        max_output_tokens = 1000
        top_p = 0.5

        # Save the PIL Image to a byte buffer
        byte_buffer = io.BytesIO()
        pil_image.save(byte_buffer, format='JPEG')
        image_data = byte_buffer.getvalue()

        data = db.generate_content(query_text, files, temperature, max_output_tokens, top_p)
        
        if data:
            db.insert_to_database(data, image_data)
            return jsonify({"message": "Image processed and saved successfully", "data": data}), 200
        else:
            return jsonify({"error": "Failed to extract data from image"}), 500
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

    
# Route to handle image upload and insertion
@app.route('/upload_face', methods=['POST'])
def upload_face():
    try:
        # Extract image data from request
        image_data = base64.b64decode(request.form['image_data'].split(",")[1])

        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()

        # Prepare SQL query
        # query = '''
        #     # INSERT INTO UserUpload (UploadedImage)
        #     # VALUES (?)
        # '''

        query = '''
            UPDATE UserUpload
            SET UploadedImage = ?
            WHERE UploadTime = (SELECT TOP 1 UploadTime FROM UserUpload ORDER BY UploadTime DESC)
        '''

        cursor.execute(query, (image_data,))
        conn.commit()

        # Close resources
        cursor.close()
        conn.close()

        return jsonify({'message': 'Image uploaded successfully'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

