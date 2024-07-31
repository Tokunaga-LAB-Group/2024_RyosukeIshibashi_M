# jsonファイルからデータを読み込む
# ついでにデータの前処理までやる

import numpy as np
import pandas as pd


# jsonデータとってくる
def readJsonRaw(filename, pattern, stim):
    """
    param filename: 読み込みたいjsonファイル
    param pattern: 読み込みたい匂い刺激パターン(今は 'p1' だけ)
    param stim: 読み込みたい匂い濃度 (-5, -6, -7, -8, -9, 0 (dtype=int))
    return: stimData, responseData, responseMean, responseVar
    """

    MAX_ID = 137

    df = pd.read_json(filename, orient="index")

    # 無理やりnumpy配列に変換
    predata = np.array(df.query('pattern == @pattern and stim == @stim')["data"].to_numpy())
    responseData = np.array([d for d in predata])

    # print(data.shape)

    # 平均と分散
    responseMean = np.mean(responseData, axis=0)
    responseVar = np.var(responseData, axis=0)

    # print(mean.shape, var.shape)


    # 匂い刺激入力データ
    df = pd.read_json("../input/input_pattern.json")
    stimList = {-5:5, -6:4, -7:3, -8:2, -9:1, 0:0}
    stimData = df["p0"].to_numpy() * stimList[stim]
    
    return stimData, responseData, responseMean, responseVar



# 読み込んだ後に値を正規化？
# jsonデータとってくる
def readJsonProcess(filename, pattern, stim, tgt=300):
    """
    param filename: 読み込みたいjsonファイル
    param pattern: 読み込みたい匂い刺激パターン(今は 'p1' だけ)
    param stim: 読み込みたい匂い濃度 (-5, -6, -7, -8, -9, 0 (dtype=int))
    param tgt: 平均を求める起点フレーム
    return: stimData, (処理後の)responseData, responseMean, responseVar
    """

    stimData, responseDataRaw, _, _  = readJsonRaw(filename, pattern, stim)

    responseData = []
    # データ処理
    for data in responseDataRaw:
        # 300~320フレームの平均が0になるようにする
        f0 = np.mean(data[tgt:tgt+20])
        responseData.append(data-f0)
    responseData = np.array(responseData)

    # 平均と分散
    responseMean = np.mean(responseData, axis=0)
    responseVar = np.var(responseData, axis=0)

    return stimData, responseData, responseMean, responseVar







if __name__ == "__main__":
    print("hoge")

    # stim, data, mean, var = readJsonRaw("../input/data_all.json", "p1", -6)
    stim, data, mean, var = readJsonProcess("../input/data_all.json", "p1", -6)

    # print(stim.shape)
    # print(data.shape)
    print(mean)
    print(var)