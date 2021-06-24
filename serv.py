import os, shutil
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import keras
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import segmentation_models as sm
from flask import Flask, request, render_template, flash, redirect, url_for
from flask_cors import CORS
from werkzeug.utils import secure_filename


from config import config
from image_processor import ImageRunner

PORT = config['port']
UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
cors = CORS(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


BACKBONE = 'efficientnetb3'
preprocess_input = sm.get_preprocessing(BACKBONE)

# define network parameters
n_classes = len(config['all_classes'])
activation = 'softmax'

#create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
model.load_weights(config['model'])


ALLOWED_EXTENSIONS = ['jpg','JPG','jpeg','JPEG','png','PNG']
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def clean_folder():
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='result/' + filename), code=301)

@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        clean_folder()
        filename = secure_filename(file.filename)

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res = ImageRunner(model, preprocess_input, img).process_image()
        
        return render_template('upload.html', filename=res)
    else:
        flash('Allowed image types are -> jpg, JPG, jpeg, JPEG, png, PNG')
        return redirect(request.url)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=False)