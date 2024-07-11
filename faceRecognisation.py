import face_recognition
import numpy
from PIL import Image
import numpy as np

def verifyFace(image1, image2):

    try:
        encoding1 = face_recognition.face_encodings(image1)[0]
        encoding2 = face_recognition.face_encodings(image2)[0]
    except IndexError:
        # If no faces are found in either of the images, return False
        print("No faces found in one of the images!")
        return False

    # Compare the faces
    results = face_recognition.compare_faces([encoding1], encoding2)

    return results[0]
