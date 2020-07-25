
from flask import Flask, request, jsonify, render_template, Response

from flask import redirect, url_for
import os
from werkzeug.utils import secure_filename

import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from browser_camera import Camera
from makeup_artist import Makeup_artist

from sys import stdout
import logging
from flask_socketio import SocketIO

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

def gen():
    while True:
        #get camera frame
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
# Load the model
mask_model = tensorflow.keras.models.load_model('keras_model.h5')

app = Flask(__name__)

app.logger.addHandler(logging.StreamHandler(stdout))
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
socketio = SocketIO(app)
camera = Camera(Makeup_artist())

@socketio.on('input image', namespace='/test')
def test_message(input):
    input = input.split(",")[1]
    camera.enqueue_input(input)
    #camera.enqueue_input(base64_to_pil_image(input))


@socketio.on('connect', namespace='/test')
def test_connect():
    app.logger.info("client connected")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/model', methods=['POST'])
def model():
    # Main page
    ml_models = [str(x) for x in request.form.values()]
    print("ml_model--->>>",ml_models)
    if ml_models[0]=='InputFile_MaskPrediction':
        return redirect(url_for("mask_detect"))
    if ml_models[0]=='Webcam_MaskPrediction':
        return redirect(url_for("stream"))

@app.route('/mask_detect',methods=['GET'])
def mask_detect():
    # InputFile page
    return render_template('mask.html')

@app.route('/stream',methods=['GET'])
def stream():
    # Webcam page
    return render_template('webindex.html')


@app.route('/webstream')
def webstream():
    '''
    For rendering results on HTML GUI
    '''
    print("web strem start opening")
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        print("filepath ->>>>",file_path)
        # Make prediction

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # Replace this with the path to your image
        image = Image.open(file_path)

        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        print("Image type is", type(image))
        image = ImageOps.fit(image, size, Image.ANTIALIAS)

        # turn the image into a numpy array
        image_array = np.asarray(image)


        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = mask_model.predict(data)
        print(prediction)


        if prediction[0][0] > prediction[0][1]:
            maskdata ='Mask On'
        else:
            maskdata = 'No Mask'
        os.remove(os.path.join(basepath, 'uploads', secure_filename(f.filename)))
        return maskdata

    return None


if __name__ == "__main__":
    app.run(debug=True)
