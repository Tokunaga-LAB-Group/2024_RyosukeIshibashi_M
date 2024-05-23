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


if __name__ == "__main__":
    print("main")

    #### make data
    # nフレーム先予測
    # ノイズの入った正弦波を予測
    T = 1000      # データ長
    diff = 100     # 予測するフレーム数
    amp = 1       # 振幅
    period = 100  # 周期
    noize = 0.1   # ノイズの大きさ

    noizeData = cp.random.uniform(-noize, noize, T+diff)
    x = cp.arange(T+diff) * (2 * cp.pi / period)
    rawData = amp * cp.sin(x)
    data = rawData + noizeData

    label = data[diff:]
    data = data[:-diff]

    trainData = data[:700]
    trainLabel = label[:700]
    testData = data[700:]
    testLabel = data[700:]

    #### layer
    nodeNum = 100

    # Input
    inputLayer = InputLayer(1, 16, inputScale=1)

    # Reservoir
    reservoirLayer = ReservoirLayer(16, nodeNum, nodeNum, 0.2, 0.95, cp.tanh, 0.9)

    # Output
    outputLayer = OutputLayer(nodeNum, 1)


    #### ESN
    
    model = ESN(inputLayer, reservoirLayer, outputLayer)

    optimizer = Tikhonov(outputLayer.inputDimention, outputLayer.outputDimention, 0.1)

    # print(outputLayer.internalConnection.shape)

    # train
    trainOutput = model.train(trainData, trainLabel, optimizer)

    # print(outputLayer.internalConnection.shape)

    # test
    model.reservoirLayer.resetReservoirState()
    testOutput = model.predict(testData)


    #### graph

    fig = plt.figure(figsize=[12, 3])

    ax1 = fig.add_subplot(1,2,1)
    ax1.set_title("Result", fontsize=20)
    ax1.plot(cp.asnumpy(trainLabel), color="k", label="Test Label", linewidth=0.9, linestyle=":")
    ax1.plot(cp.asnumpy(trainOutput), label="Test Output", alpha=0.7, linewidth=0.9)
    ax1.grid(linestyle=":")
    ax1.set_xlabel("frame")

    ax1.legend()

    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("Result", fontsize=20)
    ax2.plot(cp.asnumpy(testLabel), color="k", label="Test Label", linewidth=0.9, linestyle=":")
    ax2.plot(cp.asnumpy(testOutput), label="Test Output", alpha=0.7, linewidth=0.9)
    ax2.grid(linestyle=":")
    ax2.set_xlabel("frame")

    ax2.legend()

    # 生成するファイル名
    fname = "output/20240523/test10.png"
    # 保存
    plt.savefig(fname, bbox_inches="tight", pad_inches=0.05, dpi=400)


