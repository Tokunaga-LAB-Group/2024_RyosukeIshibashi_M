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
    return: stimData, responseData, responseMean, responseStdError
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
    # 標準誤差を求める
    responseStdError = np.std(responseData, ddof=1, axis=0) / np.sqrt(len(responseData))

    # print(mean.shape, var.shape)


    # 匂い刺激入力データ
    df = pd.read_json("../input/input_pattern.json")
    stimList = {-5:5, -6:4, -7:3, -8:2, -9:1, 0:0}
    stimData = df[pattern].to_numpy() * stimList[stim]
    
    return stimData, responseData, responseMean, responseStdError


# jsonデータとってくる
def readJsonRawUnveiled(filename, pattern, stim, type, target):
    """
    param filename: 読み込みたいjsonファイル
    param pattern: 読み込みたい匂い刺激パターン(今は 'p2' だけ)
    param stim: 読み込みたい匂い濃度 (-5, -6, -7, -8, -9, -0 (dtype=string))
    param type: 読み込みたい線虫のタイプ(N2, odr-3)
    param target: 読み込みたいデータの対象(GCaMP, paQuasAr3)
    return: stimData, responseData, responseMean, responseStdError
    """

    # MAX_ID = 646

    df = pd.read_json(filename, orient="index")

    stimInt = int(stim)
    # 無理やりnumpy配列に変換
    predata = np.array(df.query('pattern == @pattern and stim == @stimInt and type == @type and target == @target')["data"].to_numpy())
    # print(predata)
    responseData = np.array([d for d in predata])

    # print(data.shape)

    # 平均と分散
    responseMean = np.mean(responseData, axis=0)
    # responseVar = np.var(responseData, axis=0)
    # 標準誤差を求める
    responseStdError = np.std(responseData, ddof=1, axis=0) / np.sqrt(len(responseData))

    # print(mean.shape, var.shape)


    # 匂い刺激入力データ
    # pattern = "p2"
    df = pd.read_json("/home/ishibashi/Reservoir_ESN/input/input_pattern.json")
    # print(np.array(df[pattern]["data"]))
    stimList = {"-5":5, "-6":4, "-7":3, "-8":2, "-9":1, "-0":0}
    stimData = np.array(df[pattern]["data"]) * stimList[stim]
    
    return stimData, responseData, responseMean, responseStdError


# 読み込んだ後に値を正規化？
# jsonデータとってくる
def readJsonProcess(filename, pattern, stim, tgt=300):
    """
    param filename: 読み込みたいjsonファイル
    param pattern: 読み込みたい匂い刺激パターン(今は 'p1' だけ)
    param stim: 読み込みたい匂い濃度 (-5, -6, -7, -8, -9, 0 (dtype=int))
    param tgt: 平均を求める起点フレーム
    return: stimData, (処理後の)responseData, responseMean, responseStdError
    """

    stimData, responseDataRaw, _, _  = readJsonRaw(filename, pattern, stim)

    responseData = []
    # データ処理
    for data in responseDataRaw:
        # 300~320フレームの平均で全体を割る
        f0 = np.mean(data[tgt:tgt+20])
        responseData.append(data / f0) # 300~320付近が1になるように
    responseData = np.array(responseData)

    # 平均と分散
    responseMean = np.mean(responseData, axis=0)
    # responseVar = np.var(responseData, axis=0)
    # 標準誤差を求める
    responseStdError = np.std(responseData, ddof=1, axis=0) / np.sqrt(len(responseData))

    return stimData, responseData, responseMean, responseStdError


# データをとってきた後に訓練データとテストデータ作るやつ
def readJsonAll(filename, pattern, stim, tgt=300, seed=0):
    """
    param filename: 読み込みたいjsonファイル
    param pattern: 読み込みたい匂い刺激パターン(今は 'p1' だけ)
    param stim: 読み込みたい匂い濃度 (-5, -6, -7, -8, -9, 0 (dtype=int))
    param tgt: 平均を求める起点フレーム
    param seed: シャッフルするときのシード値, Noneだとシャッフルしない
    return: stimDataAll, (処理後の)responseDataAll, (テスト用)stimDataTest, responseMean, responseStdError
    """

    # 読み込む濃度のリスト
    stimList = [-5, -6, -7, -8, -9, 0]

    # データ処理
    dataAll = []
    stimDataTest = []
    responseMean = 0
    responseStdError = 0
    for sl in stimList:
        stimData, data, mean, stde = readJsonProcess(filename, pattern, sl)
        for d in data:
            # 後でシャッフルするためにタプル形式
            dataAll.append((stimData.copy(), d))

        # 今の濃度が指定された奴ならデータを保存
        if sl == stim:
            stimDataTest = stimData
            responseMean = mean
            responseStdError = stde

    # データのシャッフル
    if seed != None:
        np.random.seed(seed)
        np.random.shuffle(dataAll)

    # タプルから出す
    stimDataAll = []
    responseDataAll = []
    for (stim, data) in dataAll:
        stimDataAll.append(stim)
        responseDataAll.append(data)


    return np.array(stimDataAll), np.array(responseDataAll), np.array(stimDataTest), np.array(responseMean), np.array(responseStdError)


# trainを全データにしたやつ
def readJsonAllUnveiled(filename, pattern, stim, type, target):
    """
    param filename: 読み込みたいjsonファイル
    param pattern: 読み込みたい匂い刺激パターン(今は 'p2' だけ)
    param stim: 読み込みたい匂い濃度 (-5, -6, -7, -8, -9, -0 (dtype=string))
    param type: 読み込みたい線虫のタイプ(N2, odr-3)
    param target: 読み込みたいデータの対象(GCaMP, paQuasAr3)
    return: stimDataAll, responseDataAll, stimDataTest, responseMean, responseStdError
    """

    # testデータを取ってくる
    stimDataTest, _, responseMean, responseStdError = readJsonRawUnveiled(filename, pattern, stim, type, target)

    # 全濃度のデータを取得
    stimDataAll = []
    responseDataAll = []
    for s in ["-5", "-6", "-7", "-8", "-9", "-0"]:
        sData, sResponse, _, _ = readJsonRawUnveiled(filename, pattern, s, type, target)
        stimDataAll.extend([sData] * len(sResponse))
        responseDataAll.extend(sResponse)

    return np.array(stimDataAll), np.array(responseDataAll), np.array(stimDataTest), np.array(responseMean), np.array(responseStdError)

# 学習データからtargetを抜く
def readJsonAllUnveiled2(filename, pattern, stim, type, target):
    """
    param filename: 読み込みたいjsonファイル
    param pattern: 読み込みたい匂い刺激パターン(今は 'p2' だけ)
    param stim: 読み込みたい匂い濃度 (-5, -6, -7, -8, -9, -0 (dtype=string))
    param type: 読み込みたい線虫のタイプ(N2, odr-3)
    param target: 読み込みたいデータの対象(GCaMP, paQuasAr3)
    return: stimDataAll, responseDataAll, stimDataTest, responseMean, responseStdError
    """

    # testデータを取ってくる
    stimDataTest, _, responseMean, responseStdError = readJsonRawUnveiled(filename, pattern, stim, type, target)

    # 全濃度のデータを取得
    stimDataAll = []
    responseDataAll = []
    for s in ["-5", "-6", "-7", "-8", "-9", "-0"]:
        if s == stim: 
            continue
        sData, sResponse, _, _ = readJsonRawUnveiled(filename, pattern, s, type, target)
        stimDataAll.extend([sData] * len(sResponse))
        responseDataAll.extend(sResponse)

    return np.array(stimDataAll), np.array(responseDataAll), np.array(stimDataTest), np.array(responseMean), np.array(responseStdError)





if __name__ == "__main__":
    print("hoge")

    # stim, data, mean, var = readJsonRaw("../input/data_all.json", "p1", -6)
    stim, data, mean, stde = readJsonRawUnveiled("/home/ishibashi/Reservoir_ESN/input/data_unveiled_fig5_ab.json", "p2", "-6", "N2", "paQuasAr3")
    # stimAll, dataAll, stimTest, mean, stde = readJsonAllUnveiled("/home/ishibashi/Reservoir_ESN/input/data_unveiled_fig5_ab.json", "p2", "-6", "N2", "paQuasAr3")
    # stim, data, mean, var = readJsonProcess("../input/data_all.json", "p1", -6)
    # stimAll, dataAll, stimTest, mean, stde = readJsonAll("../input/data_all.json", "p1", -6)
    # stimAll, dataAll, stimTest, mean, stde = readJsonPert("../input/data_all.json", "p1", -6, isAll=False)

    # inputData, responseData, responseMean, responseStdError = rj.readJsonProcess(jsonFname, "p1", stim)

    print(stim.shape, data.shape)
    # print(stimAll.shape, dataAll.shape)
    # print(stimTest.shape)
    # print(mean)
    # print(stde)