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
    T = 1010      # データ長
    amp = 1       # 振幅
    period = 100  # 周期
    noize = 0.1   # ノイズの大きさ

    noizeData = cp.random.uniform(-noize, noize, T)
    x = cp.linspace(0, 2*cp.pi, period)
    rawData = amp * cp.sin(cp.resize(x, T))
    data = rawData + noizeData

    label = data[10:]
    data = data[:-10]

    trainData = data[:700]
    trainLabel = label[:700]
    testData = data[700:]
    testLbale = data[700:]

    #### layer
    nodeNum = 200

    # Input
    inputLayer = InputLayer(1, 16, inputScale=1)

    # Reservoir
    reservoirLayer = ReservoirLayer(16, 16, nodeNum, 0.2, 0.9, cp.tanh, 0.1)

    # Output
    outputLayer = OutputLayer(16, 1)


    #### ESN
    
    model = ESN(inputLayer, reservoirLayer, outputLayer)

    optimizer = Tikhonov(nodeNum, 16, 0.1)

    # train
    trainOutput = model.train(trainData, trainLabel, optimizer)

    # test
    testOutput = model.predict(testData)



