# マルチリザバー実装
# とりあえずシンプルなリザバーを2つ並列にする
# それぞれのハイパーパラメータは共通？


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
from models.model import ESN, ESNs, Tikhonov, Reservoir
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

    # リザバー層の諸々パラメータ
    parser.add_argument("--reservoir_num", type=int, default=1,
                        help="Number of Reservoir")
    parser.add_argument("--N_x", type=int, nargs="*", required=True,
                        help="Number of node in Reservoir")
    parser.add_argument("--input_scale", type=float, default=1.0,
                        help="Scaling rate of input data")
    parser.add_argument("--lamb", type=float, nargs="*", default=0.24,
                        help="Average distance of connection between nodes")
    parser.add_argument("--rho", type=float, nargs="*", required=True,
                        help="Spectral Radius setpoints")
    parser.add_argument("--leaking_rate", type=float, nargs="*", required=True,
                        help="Value of reaking late")
    parser.add_argument("--feedback_scale", type=float, default=None,
                        help="Feedback rate of reservoir state")
    parser.add_argument("--noise_level", type=float, default=None,
                        help="Noise level of input data")
    parser.add_argument("--tikhonov_beta", type=float, default=0.01,
                        help="Regularization parameter for Ridge regression")
    


    # 解析結果を返す
    return parser.parse_args()



# 出力画像生成
def makeFig(flag, model, tLabel, tData, tDataStd, tY, rmse, nrmse, viewLen=2450,):
    '''
    param flag: 画像を保存するかを管理するフラグ trueで保存
    param model: 作ったモデルのインスタンス
    param tLabel: testLabel
    param tData: testData
    param tDataStd: testDataStd
    param tY: testY
    param rmse: RMSE
    param nrmse: NRMSE
    param viewLen: プロット時のx軸の最大値
    '''


    # データの差分を取る
    diff = tData - tY # 長さ同じじゃないとバグるので注意

    # グラフ表示
    # 見える長さ
    viewLen = 2450

    # サイズ調整
    fig = plt.figure(figsize=[16, 2.0])

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.set_title("Input", fontsize=20)
    # ax1.set_yscale("log")
    ax1.plot(tLabel[-viewLen:], color='k', label="input")
    hideValue = [x * 5 + args.bias for x in args.test_value]
    hideLabel = md.makeDiacetylData(hideValue, args.test_duration).reshape(-1, 1)
    ax1.plot(hideLabel[-viewLen:], alpha=0.0)
    ax1.set_xlabel("frame")
    # plt.plot([0, int(DETAIL*testLen)], [0.5, 0.5], color='k', linestyle = ':')
    # plt.ylim(0.3, 3.3)
    # plt.xlim(500, 2000)
    # plt.legend(loc="upper right")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.set_title("Prediction", fontsize=20)
    # plt.plot(pred, label="predict")
    ax2.plot(tY[-viewLen:], color="k", label="predict")
    ax2.grid(linestyle=":")
    ax2.set_xlabel("frame")
    # plt.plot(datas[0][-2450:] * 0.01, label="data", color="gray", linestyle=":")
    # plt.xlim(500, 2000)
    # plt.legend(loc="upper right")

    ax3 = fig.add_subplot(1, 4, 3, sharey=ax2) # y軸共有
    ax3.set_title("Ground Truth", fontsize=20)
    ax3.fill_between(np.linspace(0, len(tData)-1, len(tData)), 
        (tData[-viewLen:] + tDataStd[-viewLen:]).reshape(-1), 
        (tData[-viewLen:] - tDataStd[-viewLen:]).reshape(-1), 
        alpha=0.15, color='k', label="std")
    ax3.plot(tData[-viewLen:], color="k", label="mean", linewidth=0.5)
    # 軸調整用
    ax3.set_ylim(-0.5, 1.5)
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


    if flag:

        # 生成するファイル名
        fname = args.figure_save_path + args.figure_save_name
        # csvファイルの名前
        csvFname = []
        for cfname in args.csv_filename:
            csvFname.append(args.csv_filepath + cfname)
        
        # 保存
        plt.savefig(fname, bbox_inches="tight", pad_inches=0.05, dpi=400)

        # 情報書き込み
        text = "// info start\n\n"
        text += (time.ctime(os.path.getatime(__file__)) + "\n\n")
        text += f"csv file name = {csvFname}\n"
        text += f"DATALEN = {args.data_length}\n"
        text += f"TRAIN_VALUE = {args.train_value}\n"
        text += f"TRAIN_DURATION = {args.train_duration}\n"
        text += f"TEST_VALUE = {args.test_value}\n"
        text += f"TEST_DURATION = {args.test_duration}\n"
        text += f"transLen = {args.transition_length}\n\n"
        text += ("ESN param\n")
        text += (model.info() + "\n")
        text += ("score\n")
        text += (f"RMSE = {rmse}\n")
        text += (f"NRMSE = {nrmse}\n")
        text += ("\n")

        text += ("// info end\n")

        # print(text)
        stg.stgWrite(fname, text)


    plt.show()

    # print(stg.stgRead(fname))




#################################################################################

# メイン処理
def main():
    # データ生成

    # 諸々のパラメータ
    DATALEN = args.data_length # 全体のデータ長
    BIAS = args.bias # 定常状態用
    # TRAIN_VALUE = [x + BIAS for x in [0, 1, 0]] # 値
    TRAIN_VALUE = args.train_value
    TRAIN_DURATION = args.train_duration # 継続時間
    # TEST_VALUE = [x + BIAS for x in [0, 1, 0, 1, 0]]
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
    trainPeriod = np.tile(trainPeriod, len(datas)) if trainPeriod is not None else None

    # テストデータ
    inputLabel = {"10-5":5, "10-6":4, "10-7":3, "10-8":2, "10-9":1, "0":0} # エレガントじゃない
    testData = []
    testDataStd = []
    testValue = []
    if TEST is None: # テスト対象未指定なら全部でやる
        TEST = ["10-5", "10-6", "10-7", "10-8", "10-9", "0"]
    for test in TEST:
        testData.extend(csvDatasMean[test].reshape(-1, 1))
        testDataStd.extend(csvDatasStd[test].reshape(-1, 1))
        testValue.extend([x * inputLabel[test] + BIAS for x in TEST_VALUE])
    # numpy配列へ変換
    testData = np.array(testData)
    testDataStd = np.array(testDataStd)
    testValue = np.array(testValue)
    testLabel = md.makeDiacetylData(testValue, TEST_DURATION).reshape(-1, 1)


    # モデル生成

    # リザバー層の数
    resNum = args.reservoir_num
    if resNum != len(args.N_x) or resNum != len(args.lamb) \
        or resNum != len(args.rho) or resNum != len(args.leaking_rate):
        print("The number of some parameters in the reservoir layer does not match \"reservoir_num\" !!")
        exit()

    res = []
    inputMasks = []
    outputMasks = []
    for i in range(resNum):

        # マスク指定
        N_x = args.N_x[i] # ノード数
        inputMasks.append([1 if i<32  else 0 for i in range(N_x)]) # 入力ノード(1 が有効，0 が無効)
        outputMasks.append([1 if i<128 else 0 for i in range(N_x)]) # 出力ノード

        # モデルを作る
        # Reservoir層は外部で定義するようにした
        res.append(Reservoir(N_x, args.lamb[i], args.rho[i], np.tanh, args.leaking_rate[i], seed=i+1127))



    model = ESNs(trainLabel.shape[1], trainData.shape[1], 
                reservoirs = res, 
                reservoir_num = resNum,
                input_scale = args.input_scale,
                fb_scale = args.feedback_scale,
                fb_seed = 99,
                noise_level = args.noise_level,
                input_mask = inputMasks,
                output_mask = outputMasks
                )

    
    # 学習
    outputMaskConcat = []
    for i in range(resNum):
        outputMaskConcat.extend(outputMasks[i].copy())
    trainY = model.train(trainLabel, trainData,
                        Tikhonov(N_x*2, trainData.shape[1], outputMaskConcat, args.tikhonov_beta),
                        trans_len=transLen,
                        period=trainPeriod)



    # 出力
    for res in model.Reservoirs:
        res.reset_reservoir_state()
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

    makeFig(saveFig, model, testLabel, testData, testDataStd, testY, RMSE, NRMSE)








# メイン関数
if __name__ == "__main__":
    args = getArg()
    main()