#!/usr/bin/env python
# coding: utf-8

# ### keras-retinanet 패키지를 이용하여 이미지와 영상 Object Detection 수행
# *  Pretrained된 coco 모델을 로드 하고 이를 이용하여 Object Detection 수행

# #### 관련 모듈 import 

# In[1]:


import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time
import base64
import codecs, json
import pickle
import matplotlib.pyplot as plt
import tensorflow.keras
from PIL import Image

# import keras
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
#from keras_retinanet.utils.gpu import setup_gpu
from tensorflow.keras.models import Model


# In[2]:


import tensorflow as tf


# 현 버전에서 이슈가 있어서 주석 처리합니다. 2020.08.29
#gpu = 0
#setup_gpu(gpu)


# #### keras-retinanet으로 pretrained된 coco 모델 다운로드하고 해당 모델을 로드

# In[7]:


# 아래 모델은 https://github.com/fizyr/keras-retinanet/releases 에서 download 받을 수 있음. 
# 해당 모델 h5 파일을 snapshot 디렉토리에 저장 후 retina model의 load_model()을 이용하여 모델 로딩.
# !wget https://github.com/fizyr/keras-retinanet/releases/download/0.5.1/resnet50_coco_best_v2.1.0.h5 


# In[8]:


os.path.join('./keras-retinanet', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')


# In[9]:


model_path = os.path.join('./keras-retinanet', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

# pretrained coco 모델 파일을 retinanet 모델로 로딩.  
retina_model = models.load_model(model_path, backbone_name='resnet50')


# # Feature Map 출력

# #### coco 데이터 세트의 클래스id별 클래스명 지정. 

# In[10]:


labels_to_names_seq = {0:'person',1:'bicycle',2:'car',3:'motorbike',4:'aeroplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',10:'fire hydrant',
                        11:'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',
                        21:'bear',22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',27:'tie',28:'suitcase',29:'frisbee',30:'skis',
                        31:'snowboard',32:'sports ball',33:'kite',34:'baseball bat',35:'baseball glove',36:'skateboard',37:'surfboard',38:'tennis racket',39:'bottle',40:'wine glass',
                        41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',
                        51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',57:'sofa',58:'pottedplant',59:'bed',60:'diningtable',
                        61:'toilet',62:'tvmonitor',63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'cell phone',68:'microwave',69:'oven',70:'toaster',
                        71:'sink',72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',78:'hair drier',79:'toothbrush' 
                      }



# In[13]:


# load image
#imagePath = "beatles01.jpg"
#image = read_image_bgr('../../data/image/'+imagePath)
#print('image shape:', image.shape)
# copy to draw on
#draw = image.copy()
#draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# 모델에 입력전에 이미지 사전 처리. keras-retinanet은 image
#image = preprocess_image(image)
#image, scale = resize_image(image)
#print('resized image size:', image.shape, 'scale:', scale)

#image_arr = [[image]]

# #### 원본 이미지를 Object Detection 후에 시각화 표시. 


# load image
imageName = "beatles01.jpg"
imagePath = "../../data/image/" + imageName
image = read_image_bgr(imagePath)
print('image shape:', image.shape)
# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

img = image.copy()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 모델에 입력전에 이미지 사전 처리. keras-retinanet은 image
image = preprocess_image(image)
image, scale = resize_image(image)
print('resized image size:', image.shape, 'scale:', scale)

# 이미지에 대해 Object Detection 수행. 
start = time.time()
boxes, scores, labels = retina_model.predict_on_batch(np.expand_dims(image, axis=0))
print(boxes.shape, scores.shape, labels.shape)
print("processing time: ", time.time() - start)

# correct for image scale
boxes /= scale

# JSON 출력 결과를 저장할 배열
JsonObject = []

# Detect한 이미지를 Crop하여 저장할 배열
CroppedImage = []

# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break
    
    b = box.astype(int)
    
    # JSON 출력 내용 입력
    JsonObject.append(
            {
                "objectID":int(label),
                "location":{
                    "xmin":float(b[0]),
                    "ymin":float(b[1]),
                    "xmax":float(b[2]),
                    "ymax":float(b[3])
                },
                "tag":str(labels_to_names_seq[label]),
                "reducedVector":["수납장 객체의 feature map"],
                "vectorShape":["feature map의 shape"]
            })
        
    # 객체부분을 Crop하여 배열에 추가
    object_img = img[b[1]:b[3],b[0]:b[2]]
    CroppedImage.append(object_img)
    #print(object_img)
    
    color = label_color(label)

    #plt.figure()
    #plt.axis('off')
    #plt.imshow(object_img)
    #plt.show()
    
    draw_box(draw, b, color=color)
    
    caption = "{} {:.3f}".format(labels_to_names_seq[label], score)
    draw_caption(draw, b, caption)

#JsonObject = JsonObject.tolist()

#print(JsonObject)

JSON_dict={"object":JsonObject,"IMG_URL":"이미지URL"}

#print(JSON_dict)
 

# 배열을JSON 파일로 dumps
# indent=4로 설정하여 보기 좋게 만듬
# JSON 파일명은 경로명 + .json(파일 서식)
textFile = imagePath + ".json"
textJSON = json.dumps(JSON_dict, indent=4)
with open(textFile, 'w') as f:
    f.write(textJSON)


    
plt.figure(figsize=(14, 14))
plt.axis('off')
plt.imshow(draw)
plt.show()


# In[23]:


#plt.figure(figsize=(14, 14))
plt.axis('off')
for i in CroppedImage:
    #print(i)
    plt.imshow(i)
    plt.show()


# In[25]:


with open('CroppedImage.txt', 'wb') as f:
    pickle.dump(CroppedImage, f)


# In[26]:


with open("CroppedImage.txt", "rb") as objFile:
    UnpickledObject = pickle.load(objFile)


# In[27]:


#plt.figure(figsize=(14, 14))
plt.axis('off')
for i in UnpickledObject:
    #print(i)
    plt.imshow(i)
    plt.show()