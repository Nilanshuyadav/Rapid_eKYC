from flask import Flask, request, render_template, jsonify
import dbController as db
import base64
import pyodbc
import numpy as np
import cv2
from flask_cors import CORS

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

@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')

@app.route("/blog", methods=['GET'])
def blog():
    return render_template('blog.html')

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

# @app.route("/fetchData", methods=['POST'])
# def fetchData():
#     try:
#         captured_image = request.form['capturedImage']
#         image_data = base64.b64decode(captured_image.split(",")[1])
#         userinfo = db.fetchData(image_data)
#         return jsonify({"message": "Image processed successfully", "userinfo": userinfo}), 200
#     except Exception as e:
#         print(f"Error processing image: {e}")
#         return jsonify({"error": "Internal Server Error"}), 500

# @app.route("/verifyFace", methods=['GET'])
# def verifyFace():
#     try:
#         image = request.files['userImage']
#         data = image.stream.read()
#         return db.compareFace(face1, face2)
#     except Exception as e:
#         print(f"Error verifying face: {e}")
#         return jsonify({"error": "Internal Server Error"}), 500

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
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
        
        data = db.extractData(image)

        conn = get_db_connection()
        cursor = conn.cursor()

        # Insert extracted data into the database
        cursor.execute("""
            INSERT INTO UserUpload (AadhaarNumber, DateOfBirth, Name, Gender, UploadedImage)
            VALUES (?, ?, ?, ?, ?)
        """, (data['Adhaar Number'], data['Date of Birth'], data['Name'], data['Sex'], image_data))
        conn.commit()

        cursor.close()
        conn.close()

        return jsonify({"message": "Image processed and saved successfully", "data": data}), 200
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(debug=True)
