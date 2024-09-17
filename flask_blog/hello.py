from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import RGBimage
import os
from DrBulbaModel import inputs
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def getValue():
    uploaded_file = request.files['input-file']

    if uploaded_file.filename != '':
        upload_folder = os.path.join(os.getcwd(), 'static', 'img')
        os.makedirs(upload_folder, exist_ok=True)
        uploaded_file.save(os.path.join(upload_folder, secure_filename(uploaded_file.filename)))
        
        global image_path
        image_path = os.path.join('static', 'img', secure_filename(uploaded_file.filename))
        print(image_path)
        global rgb
        rgb = RGBimage.getRGB(image_path)
        print(rgb)
        
        model = joblib.load('DrBulbaModel/Gabe_model')
        global response_image
        response_image = inputs.test_image(model, image_path)
        print(response_image)

        return response()
    else:
        return "No file uploaded"

@app.route('/response')
def response():
    return render_template('response.html', my_variable=response_image, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
