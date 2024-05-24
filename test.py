# なんかいろいろテスト用

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
# import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.model3 import InputLayer, ReservoirLayer, OutputLayer, ESN, Tikhonov
from orglib import make_dataset as md
from orglib import stegano as stg
from orglib import read_csv as rc


# nフレーム先予測
# ノイズの入った正弦波を予測
def nFramePredict(T, trainLen, diff, amp, period, noise):
    '''
    param T: データ長
    param trainLen: Tのうち何フレームを学習に使うか
    param diff: 予測するフレーム数
    param amp: 振幅
    param period: 周期
    param noise: ノイズの大きさ
    return: (trainGT, trainInput, testGT, testInput)
    '''

    noiseData = cp.random.uniform(-noise, noise, T+diff)
    x = cp.arange(T+diff) * (2 * cp.pi / period)
    rawData = amp * cp.sin(x)
    data = rawData + noiseData

    gt = data[diff:]
    input = data[:-diff]

    trainGT = gt[:trainLen]
    trainInput = input[:trainLen]
    testGT = gt[trainLen:]
    testInput = input[trainLen:]

    return (trainGT, trainInput, testGT, testInput)


# nフレーム先予測
# 正弦波が入った正弦波を予測
def nFramePredict2(T, trainLen, diff, amp, period, namp):
    '''
    param T: データ長
    param trainLen: Tのうち何フレームを学習に使うか
    param diff: 予測するフレーム数
    param amp: 振幅
    param period: 周期
    param namp: ノイズ代わりの正弦波の大きさ(周期は元の 1/10)
    return: (trainGT, trainInput, testGT, testInput)
    '''

    x = cp.arange(T+diff) * (2 * cp.pi / (period/10))
    noiseData = namp * cp.sin(x)
    x = cp.arange(T+diff) * (2 * cp.pi / period)
    rawData = amp * cp.sin(x)
    data = rawData + noiseData

    gt = data[diff:]
    input = data[:-diff]

    trainGT = gt[:trainLen]
    trainInput = input[:trainLen]
    testGT = gt[trainLen:]
    testInput = input[trainLen:]

    return (trainGT, trainInput, testGT, testInput)

# 周波数予測
# 数値に合わせた周波数の正弦波を出力
def fSinPredict(T, trainLen, noise):
    '''
    param T: データ長
    param trainLen: Tのうち何フレームを学習に使うか
    param noise: ノイズの大きさ
    return: (trainGT, trainInput, testGT, testInput)
    '''

    sf = 100 # サンプリングレート
    changeTarm = 100 # 周波数変化の平均値

    # 変化する周波数の最大最小
    fmax = 3.0
    fmin = 0.5

    # 信号の長さ
    sec = T / sf

    # 変化周波数設定

    # サンプリング点設定
    t = cp.arange(0, sec, 1/sf)

    # データ設定
    gt = cp.sin(2*cp.pi*1*t)
    input = cp.full(T, 1)

    trainGT = gt[:trainLen]
    trainInput = input[:trainLen]
    testGT = gt[trainLen:]
    testInput = input[trainLen:]

    return (trainGT, trainInput, testGT, testInput)


if __name__ == "__main__":
    print("main")

    #### make data
    # trainGT, trainInput, testGT, testInput = nFramePredict(1000, 700, 30, 1, 100, 0.1)
    trainGT, trainInput, testGT, testInput = nFramePredict2(1000, 700, 30, 1, 100, 0.1)
    # trainGT, trainInput, testGT, testInput = fSinPredict(1000, 700, 0)
    
    #### layer
    nodeNum = 200

    # Input
    inputLayer = InputLayer(1, 16, inputScale=1)

    # Reservoir
    reservoirLayer = ReservoirLayer(16, 32, nodeNum, 0.2, 0.95, cp.tanh, 0.9)

    # Output
    outputLayer = OutputLayer(32, 1)


    #### ESN
    
    model = ESN(inputLayer, reservoirLayer, outputLayer)

    optimizer = Tikhonov(outputLayer.inputDimention, outputLayer.outputDimention, 0.1)

    # print(outputLayer.internalConnection.shape)

    # train
    trainOutput = model.train(trainInput, trainGT, optimizer)

    # print(outputLayer.internalConnection.shape)

    # test
    model.reservoirLayer.resetReservoirState()
    testOutput = model.predict(testInput)


    #### graph

    fig = plt.figure(figsize=[12, 3])

    ax1 = fig.add_subplot(1,2,1)
    ax1.set_title("Train", fontsize=15)
    ax1.plot(cp.asnumpy(trainGT), color="k", label="GT", linewidth=0.9, linestyle=":")
    ax1.plot(cp.asnumpy(trainOutput), label="Output", alpha=0.7, linewidth=0.9)
    ax1.grid(linestyle=":")
    ax1.set_xlabel("frame")

    ax1.legend()

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("Test", fontsize=15)
    ax2.plot(cp.asnumpy(testGT), color="k", label="GT", linewidth=0.9, linestyle=":")
    ax2.plot(cp.asnumpy(testOutput), label="Output", alpha=0.7, linewidth=0.9)
    ax2.grid(linestyle=":")
    ax2.set_xlabel("frame")

    ax2.legend()

    # 生成するファイル名
    fname = "output/20240524/test05.png"
    # 保存
    plt.savefig(fname, bbox_inches="tight", pad_inches=0.05, dpi=400)


    # info表示できるか
    # print(model.info())


