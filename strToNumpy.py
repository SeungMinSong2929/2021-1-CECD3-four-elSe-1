from tqdm import tqdm
import numpy as np

def strToNumpy(df):
    for i in tqdm(range(len(df.index))):
        arr = df["IMG"][i].replace("[","").replace("]","").replace(",","").split(" ")
        npArr = np.array(list(map(int, arr))).reshape(550,550,3)
        df["IMG"][i] = npArr