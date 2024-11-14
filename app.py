import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the trained SVM model
model = joblib.load('leaf_classifier.pkl')

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Helper function to process the uploaded image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))  # Resize to match training data
    img = img.flatten()  # Flatten the image to a 1D array
    return np.array([img])

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling image upload and classification
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess and classify the image
        processed_image = preprocess_image(filepath)
        prediction = model.predict(processed_image)[0]
        result = 'Healthy' if prediction == 0 else 'Unhealthy'
        
        return jsonify({'result': result, 'image_url': url_for('static', filename=f'uploads/{filename}')})

    flash('Invalid file type')
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
