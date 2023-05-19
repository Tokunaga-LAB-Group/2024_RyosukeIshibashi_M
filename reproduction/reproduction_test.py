# 動作確認用

import sys
import pathlib
# 実行ファイルのあるディレクトリの絶対パスを取得
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append( str(current_dir) + "\..\\" )
import argparse
import os
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.model import ESN, Tikhonov
from orglib import make_dataset as md
from orglib import stegano as stg
from orglib import read_csv as rc



# シェルからの入力を解析 多分シェルに限らず入力があってたら動く
def getArg():
    # parser用意
    parser = argparse.ArgumentParser(description="ESN Reproduction")

    # 入力解析とデータ格納
    parser.add_argument("--csv_fliename", type=list(str), required=True, 
                        help="List of CSV file name.")
    parser.add_argument("--data_length", type=int, required=True,
                        help="Length of a data")
    parser.add_argument("--bias", type=float, default=0.1,
                        help="Data bias")
    parser.add_argument("--train_value", type=list(float), required=True,
                        help="List of train data values")
    parser.add_argument("--train_duration", type=list(int), required=True,
                        help="List of train data duration")
    parser.add_argument("--test_value", type=list(float), required=True,
                        help="List of test data values")
    parser.add_argument("--tset_duration", type=list(int), required=True,
                        help="List of test data duration")
    parser.add_argument("--transition_length", type=int, default=None,
                        help="Length of transition period")
    parser.add_argument("--train_period_value", type=list(int), default=None,
                        help="List of train section. Value 0 is not train to model")
    parser.add_argument("--train_period_duration", type=list(int), default=None,
                        help="List of train section duration")
    parser.add_argument("--test_name", type=list(str), default=None,
                        help="List of test data name that for training")
    parser.add_argument("--figure_save_path", type=str, default=None,
                        help="Path name for saving. No assignment to not saving")
    parser.add_argument("--figure_save_name", type=str, default=None,
                        help="Figuer name to result. No assignment to not saving")


    # 解析結果を返す
    return parser.parse_args()



# メイン処理
def main():
    # データ生成

    # 諸々のパラメータ
    DATALEN = args.data_len # 全体のデータ長
    BIAS = args.bias # 定常状態用
    # TRAIN_VALUE = [x + BIAS for x in [0, 1, 0]] # 値
    TRAIN_VALUE = args.train_value
    TRAIN_DURATION = args.train_duration # 継続時間
    # TEST_VALUE = [x + BIAS for x in [0, 1, 0, 1, 0]]
    TEST_VALUE = args.test_value
    TEST_DURATION = args.test_duration
    transLen = args.transtion_length
    if args.train_period_value is None:
        trainPeriod = None
    else:
        periodValue = args.train_period_value
        periodDura = args.train_period_duration
        trainPeriod = md.makeSquareWave(periodValue, periodDura)
    csvFname = args.csv_filename
    TEST = args.testdata # ここ関連は要修正

    if args.figure_save_path is None:
        saveFig = False # 出力を保存するか否か
    else:
        saveFig = True
        fname = args.figure_save_path + args.figure_save_name # 生成するファイル名


    # 訓練データ
    trainData = []
    trainLabel = []
    csvData, inputData, csvDatasMean, csvDatasStd = rc.readCsvAll(csvFname, 300)
    datas = []
    for data, input in zip(csvData, inputData): # 全データを一つにしてみる
        datas.append(data.copy())
        trainData.extend(data.copy())
        value = [x * input + BIAS for x in TRAIN_VALUE]
        trainLabel.extend(md.makeDiacetylData(value, TRAIN_DURATION))
    trainData = np.array(trainData).reshape(-1, 1)
    trainLabel = np.array(trainLabel).reshape(-1, 1)
    trainPeriod = np.tile(trainPeriod, len(datas))

    # テストデータ
    inputLabel = {"10-5":1e4, "10-6":1e3, "10-7":1e2, "10-8":1e1, "10-9":1e0, "0":0} # エレガントじゃない
    testValue = [x * inputLabel[TEST] + BIAS for x in TEST_VALUE]
    testLabel = md.makeDiacetylData(testValue, TEST_DURATION).reshape(-1, 1)



    # モデル生成
    N_x = 400 # ノード数
    inputMusk  = [1 if i<32  else 0 for i in range(N_x)] # 入力ノード(1 が有効，0 が無効)
    outputMusk = [1 if i<256 else 0 for i in range(N_x)] # 出力ノード
    # lambda:0.240 ~ density:0.05
    # lambda:0.350 ~ density:0.10
    # lambda:0.440 ~ density:0.15
    model = ESN(trainLabel.shape[1], trainData.shape[1], N_x, lamb=0.24,
                input_scale=1, 
                rho=0.9,
                # fb_scale=1e-3,
                fb_seed=99,
                leaking_rate=0.1,
                # noise_level=0.1,
                input_musk=inputMusk,
                output_musk=outputMusk)


    # 学習
    trainY = model.train(trainLabel, trainData,
                        Tikhonov(N_x, trainData.shape[1], outputMusk, 0.01),
                        trans_len=transLen,
                        period=trainPeriod)



    # 出力
    model.Reservoir.reset_reservoir_state()
    testY = model.predict(testLabel)

    
    # 評価
    # 最初の方を除く
    RMSE = np.sqrt(((testData[200:] - testY[200:]) ** 2).mean())
    NRMSE = RMSE/np.sqrt(np.var(testData[200:]))
    print('RMSE =', RMSE)
    print('NRMSE =', NRMSE)

    # R2 = testDataset.dataR2(testY)
    # print("R2 =", R2)

    # フリーラン
    # model.Reservoir.reset_reservoir_state()
    # testY = model.run(testLabel)

    # データの差分を取る
    diff = testData - testY # 長さ同じじゃないとバグるので注意


    # グラフ表示
    # 見える長さ
    viewLen = 2450

    # サイズ調整
    fig = plt.figure(figsize=[16, 2.0])

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.set_title("Input", fontsize=20)
    ax1.set_yscale("log")
    ax1.plot(testLabel[-viewLen:], color='k', label="input")
    hideValue = [x * 1e4 + BIAS for x in TEST_VALUE]
    hideLabel = md.makeDiacetylData(hideValue, TEST_DURATION).reshape(-1, 1)
    ax1.plot(hideLabel[-viewLen:], alpha=0.0)
    ax1.set_xlabel("frame")
    # plt.plot([0, int(DETAIL*testLen)], [0.5, 0.5], color='k', linestyle = ':')
    # plt.ylim(0.3, 3.3)
    # plt.xlim(500, 2000)
    # plt.legend(loc="upper right")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.set_title("Prediction", fontsize=20)
    # plt.plot(pred, label="predict")
    ax2.plot(testY[-viewLen:], color="k", label="predict")
    ax2.grid(linestyle=":")
    ax2.set_xlabel("frame")
    # plt.plot(datas[0][-2450:] * 0.01, label="data", color="gray", linestyle=":")
    # plt.xlim(500, 2000)
    # plt.legend(loc="upper right")

    ax3 = fig.add_subplot(1, 4, 3, sharey=ax2) # y軸共有
    ax3.set_title("Ground Truth", fontsize=20)
    ax3.fill_between(np.linspace(0, len(testData)-1, len(testData)), 
        (testData[-viewLen:] + testDataStd[-viewLen:]).reshape(-1), 
        (testData[-viewLen:] - testDataStd[-viewLen:]).reshape(-1), 
        alpha=0.15, color='k', label="std")
    ax3.plot(testData[-viewLen:], color="k", label="mean", linewidth=0.5)
    ax3.grid(linestyle=":")
    # for data in datas:
    #     ax3.plot(data.reshape(-1,1), linewidth=0.5)
    ax3.set_xlabel("frame")
    # plt.plot([0, int(DETAIL*testLen)], [0.5, 0.5], color='k', linestyle = ':')
    # plt.ylim(0.3, 3.3)
    # plt.xlim(500, 2000)
    ax3.legend(loc="upper right")


    ax4 = fig.add_subplot(1, 4, 4, sharey=ax2)
    ax4.set_title("Difference", fontsize=20)
    # ax4.set_yscale("log")
    ax4.plot(diff[-viewLen:], color='k', label="diff", linewidth=0.5)
    ax4.grid(linestyle=":")
    ax4.set_xlabel("frame")


    # plt.subplot(3, 1, 3)
    # plt.plot(pred[:, 1], label="predict")
    # plt.plot(testLabel[:, 1], color='gray', label="label", linestyle=":")
    # # plt.ylim(0.3, 3.3)
    # plt.legend()


    if saveFig:

        # 保存
        plt.savefig(fname, bbox_inches="tight", pad_inches=0.05, dpi=400)

        # 情報書き込み
        text = "// info start\n\n"
        text += (time.ctime(os.path.getatime(__file__)) + "\n\n")
        text += f"csv file name = {csvFname}\n"
        text += f"DATALEN = {DATALEN}\n"
        text += f"TRAIN_VALUE = {TRAIN_VALUE}\n"
        text += f"TRAIN_DURATION = {TRAIN_DURATION}\n"
        text += f"TEST_VALUE = {TEST_VALUE}\n"
        text += f"TEST_DURATION = {TEST_DURATION}\n"
        text += f"transLen = {transLen}\n\n"
        text += ("ESN param\n")
        text += (model.info() + "\n")
        text += ("score\n")
        text += (f"RMSE = {RMSE}\n")
        text += (f"NRMSE = {NRMSE}\n")
        text += ("\n")

        text += ("// info end\n")

        # print(text)
        stg.stgWrite(fname, text)


    plt.show()

    # print(stg.stgRead(fname))


    # csv形式で保存
    if False:

        # ファイル名生成

        # train
        fname1 = "../output/csv_data/train/" + "target_700_N2_300" + ".csv" # 目標データ
        fname2 = "../output/csv_data/train/" + "input_700_liner" + ".csv" # 入力データ
        fname3 = "../output/csv_data/train/" + "output_700_liner" + ".csv" # 出力データ

        # test
        fname11 = "../output/csv_data/test/" + "input_700_" + TEST + "_liner" + ".csv" # 入力データ
        fname12 = "../output/csv_data/test/" + "output_700_" + TEST + "_liner" + ".csv" # 出力データ
        fname13 = "../output/csv_data/test/" + "GT_700" + ".csv" # GT


        # 全部一度に書きこみ
        
        # train/目標データ
        if(not os.path.exists(fname1)): # ファイルが存在しなければ
            # ファイル開く
            F = open(fname1, "w", newline="")

            writer = csv.writer(F) #ファイルオブジェクトをcsv.writerオブジェクトに変換
            writer.writerow(trainData.reshape(-1)) #行追加

            # ファイル閉じる
            F.close()

        # train/入力データ
        if(not os.path.exists(fname2)): # ファイルが存在しなければ
            # ファイル開く
            F = open(fname2, "w", newline="")

            writer = csv.writer(F) #ファイルオブジェクトをcsv.writerオブジェクトに変換
            writer.writerow(trainLabel.reshape(-1)) #行追加

            # ファイル閉じる
            F.close()

        # train/出力データ
        if(not os.path.exists(fname3)): # ファイルが存在しなければ
            # ファイル開く
            F = open(fname3, "w", newline="")

            writer = csv.writer(F) #ファイルオブジェクトをcsv.writerオブジェクトに変換
            writer.writerow(trainY.reshape(-1)) #行追加

            # ファイル閉じる
            F.close()


        # test/入力データ
        if(not os.path.exists(fname11)): # ファイルが存在しなければ
            # ファイル開く
            F = open(fname11, "w", newline="")

            writer = csv.writer(F) #ファイルオブジェクトをcsv.writerオブジェクトに変換
            writer.writerow(testLabel.reshape(-1)) #行追加

            # ファイル閉じる
            F.close()

        # test/出力データ
        if(not os.path.exists(fname12)): # ファイルが存在しなければ
            # ファイル開く
            F = open(fname12, "w", newline="")

            writer = csv.writer(F) #ファイルオブジェクトをcsv.writerオブジェクトに変換
            writer.writerow(testY.reshape(-1)) #行追加

            # ファイル閉じる
            F.close()

        # test/GT
        if(not os.path.exists(fname13)): # ファイルが存在しなければ
            # ファイル開く
            F = open(fname13, "w", newline="")

            writer = csv.writer(F) #ファイルオブジェクトをcsv.writerオブジェクトに変換
            writer.writerow(trainData.reshape(-1)) #行追加

            # ファイル閉じる
            F.close()




if __name__ is "__main__":
    args = getArg()
    main()