# csvファイルからデータを読み込む
# ついでにデータの前処理までやる

import numpy as np
import csv



# csvファイルからデータを読み込む
def readCsv(filename):

    # ファイル開く
    F = open(filename, "r")

    #ファイルからデータを読み込み
    rows = csv.reader(F, quoting=csv.QUOTE_NONNUMERIC)
    
    # for文で行を1つずつ取り出す
    data = []
    for row in rows: 
        data.append(row)
        # print(row) # rowの中身を表示

    # ファイル閉じる
    F.close()

    return np.array(data)

# 読み込んだ後に値を正規化？
def readCsvWithCorrection(filename, tgt):
    """
    param filename: ファイル名
    param tgt: Fを参照する時刻
    return: 前処理したデータ
    """


    # データ読み込み
    rows = readCsv(filename)
    
    # for文で行を1つずつ取り出す
    data = []
    for row in rows:
        # データ補正
        # 値正規化？
        f0 = np.mean(row[tgt:tgt+20])
        ff0 = np.mean(row) / f0
        row *= ff0
        # f0地点をを0に
        row -= 1

        data.append(row)
        # print(row) # rowの中身を表示


    return np.array(data)




# 一度にデータ読み込んでついでに値も渡す
def readCsvAll(fnames, tgt, seed=917):
    """
    param filename: ディレクトリを含むファイル名のリスト
    param tgt: Fを参照する時刻
    param seed: データシャッフル時のシード値 Noneを与えるとシャッフルしない
    return: 前処理したデータ，入力値，それぞれの入力値ごとの平均，標準偏差
    """
    csvDatas = np.empty([0, 700])
    csvDatasMean = {}
    csvDatasStd = {}
    inputDatasLabel = {"10-5":5, "10-6":4, "10-7":3, "10-8":2, "10-9":1, "0":0} # エレガントじゃない
    inputDatas = np.empty(0)
    for fname in fnames:
        data = readCsvWithCorrection(fname, tgt)
        csvDatas = np.append(csvDatas, data, axis=0)
        csvDatasMean[fname.split("_")[1]] = np.mean(data, axis=0)
        csvDatasStd[fname.split("_")[1]] = np.std(data, axis=0) / np.sqrt(len(data))
        inputDatas = np.append(inputDatas, [inputDatasLabel[fname.split("_")[1]]] * len(data))
    # print(csvDatas.shape)
    # print(inputDatas.shape, inputDatas)
    # print(inputDatasLabel)
    # print(csvDatasMean)

    # 対応関係を保ったままシャッフル
    if seed != None:
        np.random.seed(seed)
        index = np.random.permutation(len(inputDatas))
    else:
        index = np.arange(len(inputDatas))
    csvDatas = csvDatas[index]
    inputDatas = inputDatas[index]

    # 返り値
    return csvDatas, inputDatas, csvDatasMean, csvDatasStd