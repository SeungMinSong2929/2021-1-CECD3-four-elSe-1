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
from tqdm import tqdm
import pickle


def imageStore(cPath, dPath):
    df = pd.read_csv(cPath+".csv")
    for label in df["LV2"].unique():
        if not os.path.isdir(cPath):
            os.makedirs(cPath)

    for idx in df.index:
        df["LV0"][idx] = df["LV0"][idx].replace("/", "+")
        df["LV1"][idx] = df["LV1"][idx].replace("/", "+")
        df["LV2"][idx] = df["LV2"][idx].replace("/", "+")

    df.to_csv(cPath+".csv", index=False, encoding="utf-8-sig")

    dropped_indices = []
    imgs = []
    lv2_names = []
    for idx in tqdm(df.index):
        try:
            res = request.urlopen(df["IMG_URL"][idx])
            img = Image.open(BytesIO(res.read()))
            if np.array(img).shape == (550, 550, 3):
                imgs.append(img)
                lv2_name = df["LV2"][idx]
                lv2_names.append(lv2_name)
            else:
                dropped_indices.append(idx)

        except:
            dropped_indices.append(idx)

    with open(dPath+"/imgs.p", "wb") as f:
        pickle.dump(imgs, f)

    with open(dPath+"/lv2_names.p", "wb") as f:
        pickle.dump(lv2_names, f)

    with open(dPath+"/dropped_indices.p", "wb") as f:
        pickle.dump(dropped_indices, f)

    with open(dPath+"/dropped_indices.p", "rb") as f:
        dropped_indices = pickle.load(f)

    with open(dPath+"/imgs.p", "rb") as f:
        imgs = pickle.load(f)

    with open(dPath+"/lv2_names.p", "rb") as f:
        lv2_names = pickle.load(f)

    df2 = df.copy()
    if len(df2["IMG_URL"]) != len(imgs):
        df2 = df2.drop([df2.index[idx] for idx in dropped_indices])
        df2.to_csv(cPath+".csv", index=False, encoding="utf-8-sig")

    for label in df["LV2"].unique():
        if not os.path.isdir(cPath+f"/{label}"):
            os.makedirs(cPath+f"/{label}")

    for idx, img in tqdm(enumerate(imgs)):
        save_path = cPath+f"/{lv2_names[idx]}"
        # print(save_path)
        img.save(save_path+f"/{df['PROD_ID'][idx]}.png")
