# データを見える化

import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd


# ファイルパスとファイル名(複数可)
FILEPATH="../input/"
FILENAME="data_all.json"


def makeImg(data, title, mean, var, fname):
    # サイズ調整
    fig = plt.figure(figsize=[10, 4])

    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title(title, fontsize=20)
    ax1.plot(data, linewidth=0.7, label="data")
    ax1.grid(linestyle=":")
    ax1.set_xlabel("frame")

    ax1.fill_between(np.linspace(0, len(data)-1, len(data)), 
        (mean + var).reshape(-1), (mean - var).reshape(-1), 
        alpha=0.15, color='k', label="var")
    ax1.plot(mean, color="k", label="mean", linewidth=0.5)

    ax1.legend()

    # 保存
    plt.savefig(fname, bbox_inches="tight", pad_inches=0.05, dpi=400)

    plt.close()



if __name__ == "__main__":

    filename = FILEPATH + FILENAME

    maxID = 137
    pattern = "p0"

    for ID in range(1, maxID):
        # データを持ってくる
        df = pd.read_json(filename, orient="index")
        data = np.array(df.query('index == @ID')["data"].to_numpy()[0])
        stim = int(df.query("index == @ID")["stim"].iloc[0])
        # print(data)

        # 平均と分散を求める
        stimData = df.query('stim == @stim')

        # print(stimData["data"].head(5))

        arr = stimData["data"].to_numpy()

        arrData = []
        for d in arr:
            arrData.append(d)
        arrData = np.array(arrData)

        # print(arrData.shape, np.mean(arrData, axis=0).shape)
        mean = np.mean(arrData, axis=0)
        var = np.var(arrData, axis=0)


        # グラフタイトル作る
        title = f"{ID}_{pattern}_{stim if stim != 0 else '-0'}"
        print(title)

        # path作成
        imgpath = "../output/20240725/"
        imgname = f"{title}.png"

        # グラフ作成
        makeImg(data, title, mean, var, imgpath + imgname)




