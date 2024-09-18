# input/ 配下のデータを一つのjsonにする

import csv
import json
import numpy as np
import os
import pandas as pd

# ファイルパスとファイル名(複数可)
FILEDIR="/home/ishibashi/Reservoir_ESN/input/Original_Data_csv/Figure5/Figure5AB" # とりあえず今はこれだけ
# FILENAME=["data_10-5_N2_300.csv", "data_10-6_N2_300.csv", "data_10-7_N2_300.csv" ,"data_10-8_N2_300.csv", "data_10-9_N2_300.csv", "data_0_N2_300.csv"]

if __name__ == "__main__":
    print("start")
    

    fileNameList = np.asarray(os.listdir(FILEDIR))
    # print(fileNameList)

    ID = 1
    pattern = "p2"
    stim = "-5"
    type = "N2"
    target = "GCaMP"

    dictData = {}

    # for current_dir, sub_dirs, files_list in os.walk(FILEPATH): 
    #     print("hoge")


    for fname in fileNameList:
        filename = os.path.join(FILEDIR, fname)
        print(filename)

        # ファイル開く
        F = open(filename, "r")

        # 濃度設定
        stim = "-" + filename.split(".")[0][-1]
        # print(stim)

        # type設定
        type = filename.split("_")[4]
        # print(type)

        # target設定
        target = filename.split("/")[-1].split("_")[0]
        # print(target)

        #ファイルからデータを読み込み
        # rows = csv.reader(F, quoting=csv.QUOTE_NONNUMERIC)
        df = pd.read_csv(F, header=None, index_col=None)

        # # for文で行を1つずつ取り出す
        # data = []
        # for row in rows: 
        #     # データ作成(辞書)
        #     print(row)
        #     dData = {"pattern":pattern, "stim":stim, "type":type, "data":row}
        #     dictData[ID] = dData

        #     ID += 1

        # for文で列を1つずつ取り出す
        data = []
        for col in df.columns:
            colData = df[col]
            # print(colData.values)
            # データ作成(辞書)
            dData = {"pattern":pattern, "stim":stim, "type":type, "target":target, "data":list(colData.values)}
            dictData[ID] = dData

            ID += 1

        # ファイル閉じる
        F.close()



    jsonData = json.dumps(dictData, ensure_ascii=False, indent=4)
    # print(jsonData)

    # jsonファイルに記録
    with open(os.path.join("/home/ishibashi/Reservoir_ESN/input", "data_unveiled.json"), 'w') as f:
        f.write(jsonData)
