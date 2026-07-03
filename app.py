from flask import Flask, render_template, request, redirect, send_from_directory, url_for
import numpy as np
import json
import uuid
import tensorflow as tf
import cv2
import os

app = Flask(__name__)

# Load the model once at startup
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

# Keep your label list as is
label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

# Load disease information from JSON
with open("plant_disease.json",'r') as file:
    plant_disease = json.load(file)

@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')

# Efficient image processing with OpenCV
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ensure RGB
    img = cv2.resize(img, (160, 160))
    img = img.astype('float32') / 255.0  # normalize
    img = np.expand_dims(img, axis=0)  # batch dimension
    return img

# Wrap prediction in tf.function for speed
@tf.function
def model_predict_tf(img):
    return model(img)

def model_predict(image_path):
    img = extract_features(image_path)
    prediction = model_predict_tf(img)
    prediction_label = plant_disease[np.argmax(prediction.numpy())]
    return prediction_label

@app.route('/upload/', methods=['POST','GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        os.makedirs("uploadimages", exist_ok=True)
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}_{image.filename}"
        image.save(temp_name)
        print(f'{temp_name}')
        prediction = model_predict(temp_name)
        return render_template('home.html', result=True, imagepath=f'/{temp_name}', prediction=prediction)
    else:
        return redirect('/')

if __name__ == "__main__":
    # Disable eager execution for speed
    tf.config.run_functions_eagerly(False)
    app.run(debug=True)
