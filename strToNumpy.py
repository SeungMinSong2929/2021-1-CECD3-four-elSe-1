def strToNumpy(df):
    for i in range(len(df.index)):
        arr = df["IMG"][i].replace("[","").replace("]","").replace(",","").split(" ")
        npArr = np.array(list(map(int, arr))).reshape(550,550,3)
        df["IMG"][i] = npArr