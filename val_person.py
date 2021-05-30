from pycocotools.coco import COCO
import requests
import pickle
from PIL import Image
import os

# instantiate COCO specifying the annotations json path
coco = COCO('instances_val2014.json')
# Specify a list of category names of interest
catIds = coco.getCatIds(catNms=['person'])
# Get the corresponding image ids and images using loadImgs
imgIds = coco.getImgIds(catIds=catIds)
images = coco.loadImgs(imgIds)

# Save the images into a local folder
num=0
img_arr = []

for im in images:
    num+=1
    print(num)
    img_data = requests.get(im['coco_url']).content

    img = Image.open('../val2014/' + im['file_name'])
    file_type = img.format
    img.save("person"+str(num).zfill(6)+".png", "PNG")
    
    if num is 1000:
        break
