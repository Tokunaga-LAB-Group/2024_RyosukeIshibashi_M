# 動作確認用

import sys
import pathlib
# 実行ファイルのあるディレクトリの絶対パスを取得
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append( str(current_dir) + "/../" )
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
    parser.add_argument("--csv_filepath", type=str, required=True,
                        help="CSV file path")
    parser.add_argument("--csv_filename", nargs="*", type=str, required=True, 
                        help="List of CSV file name.")
    parser.add_argument("--data_length", type=int, required=True,
                        help="Length of a data")
    parser.add_argument("--bias", type=float, default=0.1,
                        help="Data bias")
    parser.add_argument("--train_value", nargs="*", type=float, required=True,
                        help="List of train data values")
    parser.add_argument("--train_duration", nargs="*", type=int, required=True,
                        help="List of train data duration")
    parser.add_argument("--test_value", nargs="*", type=float, required=True,
                        help="List of test data values")
    parser.add_argument("--test_duration", nargs="*", type=int, required=True,
                        help="List of test data duration")
    parser.add_argument("--transition_length", type=int, default=None,
                        help="Length of transition period")
    parser.add_argument("--train_period_value", nargs="*", type=int, default=None,
                        help="List of train section. Value 0 is not train to model")
    parser.add_argument("--train_period_duration", nargs="*", type=int, default=None,
                        help="List of train section duration")
    parser.add_argument("--test_name", nargs="*", type=str, default=None,
                        help="List of test data name that for training")
    parser.add_argument("--figure_save_path", type=str, default=None,
                        help="Path name for saving. No assignment to not saving")
    parser.add_argument("--figure_save_name", type=str, default=None,
                        help="Figuer name to result. No assignment to not saving")

    parser.add_argument("--N_x", type=int, required=True,
                        help="Number of node in Reservoir")
    parser.add_argument("--input_scale", type=float, default=1.0,
                        help="Scaling rate of input data")
    parser.add_argument("--lamb", type=float, default=0.24,
                        help="Average distance of connection between nodes")
    parser.add_argument("--rho", type=float, required=True,
                        help="Spectral Radius setpoints")
    parser.add_argument("--leaking_rate", type=float, required=True,
                        help="Value of reaking late")
    parser.add_argument("--feedback_scale", type=float, default=None,
                        help="Feedback rate of reservoir state")
    parser.add_argument("--noise_level", type=float, default=None,
                        help="Noise level of input data")
    parser.add_argument("--tikhonov_beta", type=float, default=0.01,
                        help="Regularization parameter for Ridge regression")
    


    # 解析結果を返す
    return parser.parse_args()



# メイン処理
def main():
    # データ生成

    # 諸々のパラメータ
    DATALEN = args.data_length # 全体のデータ長
    BIAS = args.bias # 定常状態用

    TRAIN_VALUE = args.train_value
    TRAIN_DURATION = args.train_duration # 継続時間
    
    TEST_VALUE = args.test_value
    TEST_DURATION = args.test_duration
    
    transLen = args.transition_length
    
    if args.train_period_value is None:
        trainPeriod = None
    else:
        periodValue = args.train_period_value
        periodDura = args.train_period_duration
        trainPeriod = md.makeSquareWave(periodValue, periodDura)
    csvFname = []
    for fname in args.csv_filename:
        csvFname.append(args.csv_filepath + fname)
    TEST = args.test_name

    if args.figure_save_path is None:
        saveFig = False # 出力を保存するか否か
    else:
        saveFig = True
        fname = args.figure_save_path + args.figure_save_name # 生成するファイル名


    # 訓練データ
    # trainUが入力，trainGTが正解データ
    trainGT = []
    trainU = []
    csvData, inputData, csvDatasMean, csvDatasStd = rc.readCsvAll(csvFname, 300)
    datas = []
    for data, input in zip(csvData, inputData): # 全データを一つにしてみる
        datas.append(data.copy())
        trainGT.extend(data.copy())
        value = [x * input + BIAS for x in TRAIN_VALUE]
        trainU.extend(md.makeDiacetylData(value, TRAIN_DURATION))
    trainGT = np.array(trainGT).reshape(-1, 1)
    trainU = np.array(trainU).reshape(-1, 1)
    trainPeriod = np.tile(trainPeriod, len(datas)) if trainPeriod is not None else None

    # テストデータ
    # testUが入力，testGTが正解データ
    inputU = {"10-5":1e4, "10-6":1e3, "10-7":1e2, "10-8":1e1, "10-9":1e0, "0":0} # エレガントじゃない
    testGT = []
    testGTStd = []
    testValue = []
    if TEST is None: # テスト対象未指定なら全部でやる
        TEST = ["10-5", "10-6", "10-7", "10-8", "10-9", "0"]
    for test in TEST:
        testGT.extend(csvDatasMean[test].reshape(-1, 1))
        testGTStd.extend(csvDatasStd[test].reshape(-1, 1))
        testValue.extend([x * inputU[test] + BIAS for x in TEST_VALUE])
    # numpy配列へ変換
    testGT = np.array(testGT)
    testGTStd = np.array(testGTStd)
    testValue = np.array(testValue)
    testU = md.makeDiacetylData(testValue, TEST_DURATION).reshape(-1, 1)



    # モデル生成
    N_x = args.N_x # ノード数
    inputMask  = [1 if i<32  else 0 for i in range(N_x)] # 入力ノード(1 が有効，0 が無効)
    outputMask = [1 if i<256 else 0 for i in range(N_x)] # 出力ノード
    model = ESN(trainU.shape[1], trainGT.shape[1], N_x, lamb=args.lamb,
                input_scale=args.input_scale, 
                rho=args.rho,
                fb_scale=args.feedback_scale,
                fb_seed=99,
                leaking_rate=args.leaking_rate,
                noise_level=args.noise_level,
                input_mask=inputMask,
                output_mask=outputMask)


    # 学習
    trainY = model.train(trainU, trainGT,
                        Tikhonov(N_x, trainGT.shape[1], outputMask, args.tikhonov_beta),
                        trans_len=transLen,
                        period=trainPeriod)



    # 出力
    model.Reservoir.reset_reservoir_state()
    testY = model.predict(testU)

    
    # 評価
    # 最初の方を除く
    RMSE = np.sqrt(((testGT[200:] - testY[200:]) ** 2).mean())
    NRMSE = RMSE/np.sqrt(np.var(testGT[200:]))
    print('RMSE =', RMSE)
    print('NRMSE =', NRMSE)

    # データの差分を取る
    diff = testGT - testY # 長さ同じじゃないとバグるので注意


    # グラフ表示
    # 見える長さ
    viewLen = 2450

    # サイズ調整
    fig = plt.figure(figsize=[16, 2.0])

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.set_title("Input", fontsize=20)
    ax1.set_yscale("log")
    ax1.plot(testU[-viewLen:], color='k', label="input")
    hideValue = [x * 1e4 + BIAS for x in TEST_VALUE]
    hideU = md.makeDiacetylData(hideValue, TEST_DURATION).reshape(-1, 1)
    ax1.plot(hideU[-viewLen:], alpha=0.0)
    ax1.set_xlabel("frame")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.set_title("Prediction", fontsize=20)
    ax2.plot(testY[-viewLen:], color="k", label="predict")
    ax2.grid(linestyle=":")
    ax2.set_xlabel("frame")

    ax3 = fig.add_subplot(1, 4, 3, sharey=ax2) # y軸共有
    ax3.set_title("Ground Truth", fontsize=20)
    ax3.fill_between(np.linspace(0, len(testGT)-1, len(testGT)), 
        (testGT[-viewLen:] + testGTStd[-viewLen:]).reshape(-1), 
        (testGT[-viewLen:] - testGTStd[-viewLen:]).reshape(-1), 
        alpha=0.15, color='k', label="std")
    ax3.plot(testGT[-viewLen:], color="k", label="mean", linewidth=0.5)
    ax3.grid(linestyle=":")
    ax3.set_xlabel("frame")
    ax3.legend(loc="upper right")


    ax4 = fig.add_subplot(1, 4, 4, sharey=ax2)
    ax4.set_title("Difference", fontsize=20)
    ax4.plot(diff[-viewLen:], color='k', label="diff", linewidth=0.5)
    ax4.grid(linestyle=":")
    ax4.set_xlabel("frame")


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




if __name__ == "__main__":
    args = getArg()
    main()