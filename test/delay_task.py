#!/usr/bin/env python
# -*- coding: utf-8 -*-

#################################################################
# 田中，中根，廣瀬（著）「リザバーコンピューティング」（森北出版）
# 本ソースコードの著作権は著者（田中）にあります．
# 無断転載や二次配布等はご遠慮ください．
#
# delay_task.py: 本書の図3.11に対応するサンプルコード
#################################################################

# なんかいろいろテスト用

import sys
import pathlib
# 実行ファイルのあるディレクトリの絶対パスを取得
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append( str(current_dir) + "/../" )
# import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.model3 import InputLayer, ReservoirLayer, OutputLayer, ESN, Tikhonov, ParallelReservoirLayer, SerialReservoirLayer, BothReservoirLayer




cp.random.seed(seed=0)

if __name__ == '__main__':

    # 時系列入力データ生成
    T = 500  # 長さ
    u = cp.random.rand(T,1)-0.5  # 区間[-0.5, 0.5]の乱数系列

    # 時系列出力データ生成
    delay = [4, 8, 12]  # 遅延長
    d = cp.empty((T, len(delay)))
    for k in range(len(delay)):
        for t in range(T):
            d[t, k] = u[t-delay[k]]  # 遅延系列

    # 学習用情報
    T_trans = 200  # 過渡期の長さ
    train_U = u[T_trans:T].reshape(-1, 1)
    train_D = d[T_trans:T, :].reshape(-1, len(delay))


    #### layer
    nodeNum = 100

    # Input
    inputLayer = InputLayer(train_U.shape[1], nodeNum, inputScale=1)

    # model = ESN(train_U.shape[1], train_D.shape[1], N_x, density=0.05, 
    #             input_scale=1.0, rho=0.8)

    # Reservoir
    reservoirLayer = ReservoirLayer(nodeNum, nodeNum, nodeNum, 0.2, 0.8, cp.tanh, 1.0)

    # resInput1 = InputLayer(16, 32, inputScale=1, seed=11)
    # resRes1 = ReservoirLayer(32, 48, nodeNum, 0.2, 0.95, cp.tanh, 0.22, seed=101)

    # resInput2 = InputLayer(16, 16, inputScale=1, seed=12)
    # resRes2 = ReservoirLayer(16, 16, nodeNum, 0.3, 0.95, cp.tanh, 0.22, seed=102)

    # resInput3 = InputLayer(16, 16, inputScale=1, seed=13)
    # resRes3 = ReservoirLayer(16, 64, nodeNum, 0.3, 0.95, cp.tanh, 0.22, seed=103)

    # reservoirLayer = ParallelReservoirLayer(16, 64, [(resInput1, resRes1), (resInput2, resRes2)])
    # reservoirLayer = SerialReservoirLayer(16, 64, [resRes2, resRes3], 1)
    # reservoirLayer = BothReservoirLayer(16, 64, [(resInput1, resRes1), (resInput2, resRes2)], 1)


    # Output
    outputLayer = OutputLayer(nodeNum, train_D.shape[1])



    model = ESN(inputLayer, reservoirLayer, outputLayer)

    optimizer = Tikhonov(outputLayer.inputDimention, outputLayer.outputDimention, 0.0)



    # 学習（線形回帰）
    # model.train(train_U, train_D, Tikhonov(N_x, train_D.shape[1], 0.0))
    model.train(train_U, train_D, optimizer)



    # モデル出力
    train_Y = model.predict(train_U)
    # print(train_U.shape, train_Y.shape)


    # グラフ表示用データ
    T_disp = (0, T-T_trans)
    time_axis = cp.arange(T_trans, T).get()  # 時間軸
    disp_U = train_U[T_disp[0]:T_disp[1]].get()  # 入力
    disp_D = train_D[T_disp[0]:T_disp[1], :].get()  # 目標出力
    disp_Y = train_Y[T_disp[0]:T_disp[1], :].get()  # モデル出力

    # グラフ表示
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(7, 9))
    plt.subplots_adjust(hspace=0.3)

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.text(-0.15, 1, '(a)', transform=ax1.transAxes)
    plt.plot(time_axis, disp_U[:, 0], color='gray', linestyle=':')
    plt.ylim([-0.6, 0.6])
    plt.ylabel('Input')

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.text(-0.15, 1, '(b)', transform=ax2.transAxes)
    plt.plot(time_axis, disp_D[:, 0], color='gray', linestyle=':', label='Target')
    plt.plot(time_axis, disp_Y[:, 0], color='k', label='Model')
    plt.ylim([-0.6, 0.6])
    plt.ylabel('Output (k=4)')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.text(-0.15, 1, '(c)', transform=ax3.transAxes)
    plt.plot(time_axis, disp_D[:, 1], color='gray', linestyle=':', label='Target')
    plt.plot(time_axis, disp_Y[:, 1], color='k', label='Model')
    plt.ylim([-0.6, 0.6])
    plt.ylabel('Output (k=8)')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

    ax4 = fig.add_subplot(4, 1, 4)
    ax4.text(-0.15, 1, '(d)', transform=ax4.transAxes)
    plt.plot(time_axis, disp_D[:, 2], color='gray', linestyle=':', label='Target')
    plt.plot(time_axis, disp_Y[:, 2], color='k', label='Model')
    plt.ylim([-0.6, 0.6])
    plt.ylabel('Output (k=12)')
    plt.xlabel('n')
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right')

    
    # 生成するファイル名
    fname = "../output/20240716/delay_task_03.png"
    # 保存
    plt.savefig(fname, bbox_inches="tight", pad_inches=0.05, dpi=400)

