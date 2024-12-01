import os
import joblib
import json
import cv2
import numpy as np
from .forms import ImageUploadForm
from django.shortcuts import render
from django.conf import settings
import pywt  # Ensure pywt is installed

# Load Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Modified w2d function to match training preprocessing
def w2d(img, mode='haar', level=1):
    # Convert the image to grayscale (if it's not already)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to float and normalize
    img = np.float32(img)
    img /= 255.0

    # Compute multi-level coefficients using wavedec2
    coeffs = pywt.wavedec2(img, mode, level=level)

    # Process coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0  # Zero out the approximation coefficients (LL)

    # Reconstruct the image
    img_H = pywt.waverec2(coeffs_H, mode)
    img_H *= 255  # Rescale to original intensity range
    img_H = np.uint8(img_H)

    return img_H

# Detect and crop face and eyes from the image
def detect_face_and_eyes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None  # No face detected, return None

    # Assume the first detected face is the target
    (x, y, w, h) = faces[0]
    face_region = gray[y:y+h, x:x+w]
    
    # Detect eyes within the face region
    eyes = eye_cascade.detectMultiScale(face_region)
    if len(eyes) < 2:
        return None  # Not enough eyes detected, return None

    # Crop the face region with detected eyes
    return image[y:y+h, x:x+w]

# Load the celebrity class dictionary and model
def load_model_and_dict():
    model_path = os.path.join(settings.BASE_DIR, 'models', 'saved_model.pkl')
    dict_path = os.path.join(settings.BASE_DIR, 'models', 'class_dictionary.json')

    model = joblib.load(model_path)
    with open(dict_path, 'r') as f:
        class_dict = json.load(f)

    return model, class_dict

# Image preprocessing function (like the one used for training)
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    
    # Detect and crop the face and eyes
    cropped_img = detect_face_and_eyes(img)
    if cropped_img is None:
        return None  # Return None if no valid face/eyes detected

    img_resized = cv2.resize(cropped_img, (32, 32))  # Resize to 32x32

    # Apply Wavelet transformation (haar is the wavelet type and 1 is the level)
    img_har = w2d(cropped_img, 'haar', 1)
    img_har_resized = cv2.resize(img_har, (32, 32))

    # Flatten both images and combine them vertically
    combined_img = np.vstack((img_resized.reshape(32*32*3, 1), img_har_resized.reshape(32*32, 1)))

    return combined_img  # Only return combined image

# Prediction function
def predict_image(image_path):
    # Preprocess the image (resize, wavelet transform, etc.)
    combined_img = preprocess_image(image_path)
    if combined_img is None:
        return None, None, None  # Return None if no valid face/eyes detected

    # Load the model and class dictionary
    model, class_dict = load_model_and_dict()

    # Make a prediction using the model
    prediction = model.predict(combined_img.reshape(1, -1))

    # Map prediction to a celebrity name
    predicted_class = prediction[0]
    celebrity_name = [name for name, index in class_dict.items() if index == predicted_class][0]

    # Get the prediction probabilities for each class
    prediction_probs = model.predict_proba(combined_img.reshape(1, -1))[0]

    return celebrity_name, prediction_probs, class_dict  # Return class_dict to use it for formatting

def classify_image(request):
    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded file
            image = form.cleaned_data['image']
            image_path = os.path.join(settings.MEDIA_ROOT, image.name)
            with open(image_path, 'wb+') as destination:
                for chunk in image.chunks():
                    destination.write(chunk)

            # Pass the image to your ML model for prediction
            celebrity_name, prediction_probs, class_dict = predict_image(image_path)

            if celebrity_name is None:
                return render(request, 'index.html', {
                    'form': form,
                    'error': 'Face and/or eyes were not detected in the uploaded image. Please try another image.',
                })

            # Format the probabilities to show them with celebrity names
            formatted_probs = [
                {'celebrity': name, 'probability': round(prob * 100, 2)}  # Convert to percentage
                for name, prob in zip(class_dict.keys(), prediction_probs)
            ]

            # Render the results
            return render(request, 'index.html', {
                'form': form,
                'prediction': celebrity_name,
                'formatted_probs': formatted_probs,  # Passing formatted probabilities
                'uploaded_image_url': f'/media/{image.name}'
            })

    else:
        form = ImageUploadForm()

    return render(request, 'index.html', {'form': form})
