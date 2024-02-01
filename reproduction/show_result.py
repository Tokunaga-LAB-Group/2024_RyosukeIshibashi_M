# 実行結果可視化

import sys
import pathlib
# 実行ファイルのあるディレクトリの絶対パスを取得
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append( str(current_dir) + "/../" )

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np
import glob
import re
from tqdm import tqdm
from orglib import read_stegano as rs




if __name__ == "__main__":

    # ファイル名生成
    dirname = "/home/ishibashi/Reservoir_ESN/output/20240127/"
    filenameBase = "test_N2_300_result"

    # 濃度設定 [10-5, 10-6, 10-7, 10-8, 10-9, 0]
    noudo = "10-5"

    # 実際に読むファイル名取得
    filenames = glob.glob(dirname + filenameBase + "_" + noudo + "*.png")
    # print(filenames)

    # 読む
    rawData = dict()
    nrmses = []
    for fname in tqdm(filenames):
        value = rs.readNRMSE(fname)
        rawData[fname] = value
        nrmses.append(value)
    nrmses = np.array(nrmses)

    # キーをソート
    keyData = sorted(rawData)
    # print(keyData)

    # データを成型
    data = []
    for key in keyData:
        # keySplit = key.split("/")[-1].split(".")[0].split("_")[-3:-1]
        keySplit = re.split("[._]", key)[-3:-1]
        data.append([noudo, int(re.search(r"\d+", keySplit[0]).group()), int(re.search(r"\d+", keySplit[1]).group()), rawData[key]])
    
    # dataframeに変換
    df = pd.DataFrame(data, columns=["noudo", "csv seed", "reservoir seed", "NRMSE"])
    # print(df.head(10))

    # heatmap表示
    # figureオブジェクトとaxesオブジェクトの作成
    fig = plt.figure(figsize=[8, 6])

    # グラフの作成
    data_pivot = df.pivot_table(values="NRMSE", index="csv seed", columns="reservoir seed")
    sb.heatmap(data=data_pivot, cmap='Blues', annot=True, square=True, fmt=".4g", vmax=1.0, vmin=0.3)

    # 図全体のタイトルの追加
    fig.suptitle(noudo , fontsize=20)


    # 保存
    fname = "/home/ishibashi/Reservoir_ESN/output/test/test_" + noudo + "_02.png"
    plt.savefig(fname, bbox_inches="tight", pad_inches=0.05, dpi=400)

    # グラフの表示
    plt.show()



    # print(nrmses)
    print("mean = " + str(nrmses.mean()))
    print("var  = " + str(nrmses.var()))
