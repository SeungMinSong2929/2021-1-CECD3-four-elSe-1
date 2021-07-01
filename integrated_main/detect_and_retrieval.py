import matplotlib.pyplot as plt
from glob import glob

import cv2
import os
import numpy as np
import time
import base64
import codecs, json
import pickle

from PIL import Image


# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
#from keras_retinanet.utils.gpu import setup_gpu
from keras.models import Model



# object detection
os.path.join('./keras-retinanet', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

model_path = os.path.join('./keras-retinanet', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

dataset_path = "./original_test/"
output_path = "./detected_data/detected_from_test/"


retina_model = models.load_model(model_path, backbone_name='resnet50')
os.chdir(dataset_path)
dataset_list = os.listdir(os.getcwd())
os.chdir('../')


from object_detection import object_detection
object_detection(retina_model, dataset_list, dataset_path, output_path)

from image_retrieval import image_retrieval
from tqdm import tqdm

images_path = glob(output_path+"*.jpg")

for path in tqdm(images_path):
    image_pil = Image.open(path)
    image_resized = image_pil.resize((512,512))
    image_resized.save(path)
    
# try: "simpleAE", "convAE", "vgg19" , "IncepResNet", "ResNet50v2"
#     modelName = "IncepResNet"  # try: "simpleAE", "convAE", "vgg19" , "IncepResNet", "ResNet50v2"
#     trainModel = True
#     parallel = False  # use multicore processing
image_retrieval(modelName="ResNet50v2",trainModel=True, parallel=False)