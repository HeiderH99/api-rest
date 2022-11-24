#import flask
from flask import Flask, request, jsonify
#import os
from os import getcwd
from sys import path
path.append(".")
#import config
from config.default import Config

#import kera
# Keras
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import numpy as np
import cv2

#app initialization
app = Flask(__name__)
#app config
app.config.from_object(Config)

#load model
MODEL_PATH = 'Clasificador_IDC.h5'
#loanding models
model = load_model(MODEL_PATH)

PATH_FILES = getcwd() + "/storage/"


def predecir(date):
    
    image = cv2.imread(str(date))

    image_fromarray = Image.fromarray(image, 'RGB')
    resize_image = image_fromarray.resize((224, 224))
    expand_input = np.expand_dims(resize_image,axis=0)
    input_data = np.array(expand_input)
    input_data = input_data/255

    pred = model.predict(input_data)
    if pred >= 0.5:
        return "Yes"
    else:
        return "No"

@app.post("/upload")
def upload_file():
    try:
        file = request.files['file']
        file.save(PATH_FILES + file.filename)
        return jsonify(predecir(PATH_FILES + file.filename))
    except FileNotFoundError:
        return {"failed": "okay"}
    
if __name__ == '__main__':
    app.run()    