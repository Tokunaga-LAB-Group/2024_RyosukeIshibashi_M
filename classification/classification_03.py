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
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.model3 import ESN, Tikhonov, InputLayer, ReservoirLayer, OutputLayer
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
    

    # ループ処理のためのシード値
    parser.add_argument("--csv_seed", type=int, default=0,
                        help="Seed value for shuffling CSV files")
    parser.add_argument("--reservoir_seed", type=int, default=0,
                        help="Seed value for shuffling Reservoir connection")


    # 解析結果を返す
    return parser.parse_args()


# データセット作成
def makeDataset(rawData, rawLabel, dataLen, transLen=100, suffle=None):
    '''
    param rawData: 元データ
    param rawLabel: 元データに対応するラベルデータ
    param dataLen: 一つのデータの長さ
    param transLen: 過渡期の長さ
    param suffle: データをシャッフルするときのシード値 未指定ならシャッフルしない
    return: (加工済みのデータ, 加工済みのクラスラベル)
    '''

    BIAS = args.bias # 定常状態用

    TRAIN_VALUE = args.train_value
    TRAIN_DURATION = args.train_duration # 継続時間

    processData = []
    processLabel = []
    for data, label in zip(rawData, rawLabel):
        # ラベルデータを作る
        value = [x * label + BIAS for x in TRAIN_VALUE]
        inputLabel = md.makeDiacetylData(value, TRAIN_DURATION)

        for frame in range(0, len(data)-transLen, dataLen):
            processData.append(data[frame:frame+transLen+dataLen])
            processLabel.append(inputLabel[frame:frame+transLen+dataLen])

    # 対応関係を保ったままシャッフル
    if suffle != None:
        cp.random.seed(suffle)
        index = cp.random.permutation(len(processData))
        processData = processData[index]
        processLabel = processLabel[index]
    
    processData = cp.array(processData)
    processLabel = cp.array(processLabel)

    return processData, processLabel



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


# 出力画像生成
def makeFig2(flag, model, tLabel, tData, tDataStd, tY, rmse, nrmse, viewLen=2450,):
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
    fig = plt.figure(figsize=[10, 4])

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title("Input", fontsize=20)
    # ax1.set_yscale("log")
    ax1.plot(tLabel[-viewLen:], color='k', linewidth=0.5, label="input")
    # hideValue = [x * 5 + args.bias for x in args.test_value]
    # hideLabel = md.makeDiacetylData(hideValue, args.test_duration).reshape(-1, 1)
    # ax1.plot(hideLabel[-viewLen:], alpha=0.0)
    ax1.grid(linestyle=":")
    ax1.set_xlabel("frame")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title("Output", fontsize=20)
    # plt.plot(pred, label="predict")
    ax2.plot(tY[-viewLen:], label="model")
    ax2.plot(tData[-viewLen:], color="k", label="correct", alpha=0.7, linewidth=0.7, linestyle=":")
    # ax2.set_xlim(300, 700)
    # ax2.set_ylim(-0.2, 0.8)
    ax2.grid(linestyle=":")
    ax2.set_xlabel("frame")
    ax2.legend()

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
        # trainPeriod = md.makeSquareWave(periodValue, periodDura)

    csvFname = []
    for fname in args.csv_filename:
        csvFname.append(args.csv_filepath + fname)
    TEST = args.test_name

    if args.figure_save_path is None:
        saveFig = False # 出力を保存するか否か
    else:
        saveFig = True


    # 訓練データ
    trainInput = []
    trainGT = []
    csvData, inputData, csvDatasMean, csvDatasStd = rc.readCsvAll(csvFname, 300, args.csv_seed)

    # データセット作成
    classData, classLabel = makeDataset(csvData, inputData, 100)
    trainInput = classData[:800]
    trainGT = classLabel[:800]
    testInput = classData[809:]
    testGT = classLabel[809:]


    # cupyに変換とか
    trainInput = cp.array(trainInput)
    trainGT = cp.array(trainGT)
    testInput = testInput.reshape(-1)
    testGT = testGT.reshape(-1)

    # print(testInput.shape)

    # モデル生成

    #### layer
    nodeNum = args.N_x

    # Input
    inputLayer = InputLayer(1, 128, inputScale=args.input_scale)

    # Reservoir
    reservoirLayer = ReservoirLayer(128, 256, nodeNum, args.lamb, args.rho, cp.tanh, args.leaking_rate, seed=args.reservoir_seed)

    # Output
    outputLayer = OutputLayer(256, 1)


    #### ESN
    
    model = ESN(inputLayer, reservoirLayer, outputLayer)

    optimizer = Tikhonov(outputLayer.inputDimention, outputLayer.outputDimention, args.tikhonov_beta)

    
    # 学習

    # train
    trainOutput = model.trainMini(trainInput, trainGT, optimizer, transLen=100)

    # print(outputLayer.internalConnection.shape)

    # test
    model.reservoirLayer.resetReservoirState()
    testOutput = model.predict(testInput)


    # # 出力
    # model.reservoirLayer.resetReservoirState()
    # testY = model.predict(testGT)



    # 評価
    # 最初の方を除く
    RMSE = cp.sqrt(((testGT[300:] - testOutput[300:]) ** 2).mean())
    NRMSE = RMSE/cp.sqrt(cp.var(testGT[300:]))
    print('RMSE =', RMSE)
    print('NRMSE =', NRMSE)

    # R2 = testDataset.dataR2(testY)
    # print("R2 =", R2)

    # フリーラン
    # model.Reservoir.reset_reservoir_state()
    # testY = model.run(testLabel)

    makeFig2(saveFig, model, cp.asnumpy(testInput), cp.asnumpy(testGT), cp.asnumpy(testGT), cp.asnumpy(testOutput), RMSE, NRMSE)


    # # csvファイルに記録
    # with open(args.figure_save_path + 'classification_lr_be_cs_rs_02.csv', 'a') as f:
    #     writer = csv.writer(f)
    #     stim = args.test_name[0]
    #     leak = args.leaking_rate
    #     # nodeNum = args.N_x
    #     # reservoir_num = args.reservoir_num
    #     beta = args.tikhonov_beta
    #     cs = args.csv_seed
    #     rs = args.reservoir_seed
    #     writer.writerow([stim, leak, beta, cs, rs, RMSE, NRMSE])








# メイン関数
if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"

    args = getArg()
    main()