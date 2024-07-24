# input/ 配下のデータを一つのjsonにする

import csv
import json
import numpy as np

# ファイルパスとファイル名(複数可)
FILEPATH="../input/"
FILENAME=["data_10-5_N2_300.csv", "data_10-6_N2_300.csv", "data_10-7_N2_300.csv" ,"data_10-8_N2_300.csv", "data_10-9_N2_300.csv", "data_0_N2_300.csv"]

if __name__ == "__main__":
    print("start")
    

    ID = 1
    pattern = "p1"
    stim = "-5"

    dummy = [0, 1, 2, 3, 4, 5]

    dictData = {}

    for fname in FILENAME:
        filename = FILEPATH + fname

        # ファイル開く
        F = open(filename, "r")

        # 濃度設定
        stim = "-" + filename.split("_")[1].split("-")[-1]
        # print(stim)

        #ファイルからデータを読み込み
        rows = csv.reader(F, quoting=csv.QUOTE_NONNUMERIC)

        # for文で行を1つずつ取り出す
        data = []
        for row in rows: 
            # データ作成(辞書)
            dData = {"pattern":pattern, "stim":stim, "data":row}
            dictData[ID] = dData

            ID += 1


        # ファイル閉じる
        F.close()

        print(filename)


    jsonData = json.dumps(dictData, ensure_ascii=False, indent=4)
    # print(jsonData)

    # jsonファイルに記録
    with open(FILEPATH + "data_all.json", 'w') as f:
        f.write(jsonData)
