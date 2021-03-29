import pandas as pd
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from urllib import request
import urllib
from io import BytesIO
import numpy as np
from skimage import io

csvFile_path = glob("./*.csv")[1]

df = pd.read_csv(csvFile_path)

from tqdm import tqdm
imgs = []


for i in tqdm(df.index):
    try:
        with request.urlopen(df["IMG_URL"][i]) as res:
            img = np.array(Image.open(BytesIO(res.read())))
            imgs.append(list(img.flatten()))
    except:
        imgs.append(np.nan)


df2 = df.copy()
df2["IMG"] = imgs



