import cv2
import tensorflow.keras
import threading
from PIL import Image, ImageOps
import numpy as np

from io import BytesIO

import binascii
from time import sleep
from utils import base64_to_pil_image, pil_image_to_base64

mask_model = tensorflow.keras.models.load_model('keras_model.h5')
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
labels = ['Mask ON','NO Mask']

class Camera(object):
    def __init__(self,makeup_artist):
        self.to_process = []
        self.to_output = []
        self.makeup_artist = makeup_artist

        thread = threading.Thread(target=self.keep_processing, args=())
        thread.daemon = True
        thread.start()

    def process_one(self):
        if not self.to_process:
            return

        # input is an ascii string.
        input_str = self.to_process.pop(0)

        # convert it to a pil image
        input_img = base64_to_pil_image(input_str)

        # frame = np.array(input_img)
        print("Type of input image",input_img)
        data1 = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        size = (224, 224)
        image = ImageOps.fit(input_img, size, Image.ANTIALIAS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data1[0] = normalized_image_array

        ################## where the hard work is done ############
        # output_img is an PIL image
        #image_array = cv2.resize(image_array,(300,150))
        #cv2.imshow("frame",image_array)
        # ret, jpeg = cv2.imencode('.jpg', image_array)
        # image_array = cv2.resize(jpeg, (300, 150))
        PIL_image = Image.fromarray(np.uint8(image_array)).convert('RGB').resize(size=(300, 150))
        b = BytesIO()
        PIL_image.save(b, format="jpeg")
        PIL_image = Image.open(b)
        #Image.open(BytesIO(base64.b64decode(base64_img))
        #output_img = self.makeup_artist.apply_makeup(jpeg)

        # output_str is a base64 string in ascii
        output_str = pil_image_to_base64(PIL_image)

        # convert eh base64 string in ascii to base64 string in _bytes_
        self.to_output.append(binascii.a2b_base64(output_str))

    def keep_processing(self):
        while True:
            self.process_one()
            sleep(0.01)

    def enqueue_input(self, input):
        self.to_process.append(input)

    def get_frame(self):
        while not self.to_output:
            sleep(0.05)
        return self.to_output.pop(0)