import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

import os
import time

import sys

from absl import app, flags
#from tensorflow.keras import optimizers

from core.utils import decode_cfg, load_weights
from core.dataset import Dataset
from core.callbacks import COCOEvalCheckpoint, CosineAnnealingScheduler, WarmUpScheduler
from core.utils.optimizers import Accumulative


cfg = decode_cfg("/cfgs/YOLOv4_3_7Classes.yaml")


model_type = cfg['yolo']['type']

if model_type == 'yolov4':
    from core.model.one_stage.yolov4 import YOLOv4 as Model
    from core.model.one_stage.yolov4 import YOLOLoss as Loss
    num = 251
    epochs = 200


model, eval_model = Model(cfg)


init_weight="/yolov4-obj_best.weights"
load_weights(model, init_weight)


from core.utils import decode_cfg, load_weights
from core.image import draw_bboxes, preprocess_image, postprocess_image, read_image, read_video, Shader

import time
import cv2
import numpy as np


load_weights(eval_model, init_weight)


shader = Shader(cfg['yolo']['num_classes'])
names = cfg['yolo']['names']
image_size=416


def inference(image):

    h, w = image.shape[:2]
    image = preprocess_image(image, (image_size, image_size)).astype(np.float32)
    images = np.expand_dims(image, axis=0)

    tic = time.time()
    bboxes, scores, classes, valid_detections = eval_model.predict(images)
    toc = time.time()

    bboxes = bboxes[0][:valid_detections[0]]
    scores = scores[0][:valid_detections[0]]
    classes = classes[0][:valid_detections[0]]

    # bboxes *= image_size
    _, bboxes = postprocess_image(image, (w, h), bboxes)

    return (toc - tic) * 1000, bboxes, scores, classes




st.write("""
# Detecting PPE by YOLOv4 Object Detection Algorithm
### Amirhossein Ghadiri
""")

st.text("")
st.text("")
st.text("")

st.write("""
NOTE: This algorithm is trained by 1400 labeled images and is designed particularly to detect each person and the condition of his hardhat, vest, and gloves wearing.
""")

st.text("")
st.text("")
st.text("")


#st.sidebar.header('User Input Parameters')


image = read_image("/GettyImages_xavierarnau.5c70703e85023.JFIF")


from keras.preprocessing.image import load_img
from tempfile import NamedTemporaryFile
st.set_option('deprecation.showfileUploaderEncoding', False)

image2 = st.file_uploader("Please upload a 'png', 'jpg', or 'jpeg' file.", type=["png", "jpg", "jpeg"])

temp_file = NamedTemporaryFile(delete=False)
if image2:
    temp_file.write(image2.getvalue())
    image = read_image(temp_file.name)

#if image2 == None:
#    image = image1
#else:
#    data1 = image2.read()

#    image = data1    


ms, bboxes, scores, classes = inference(image)
image = draw_bboxes(image, bboxes, scores, classes, names, shader)


st.text("")
st.text("")
st.text("")


st.write("""
## Detection Results
""")

if not image2:
    st.write("""
    ### Demo:
    """)
else:
    st.write("""
    ### Your Detection:
    """)



st.image(image)

