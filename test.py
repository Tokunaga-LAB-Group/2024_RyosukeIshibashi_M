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
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.model import ESN, Tikhonov
from orglib import make_dataset as md
from orglib import stegano as stg
from orglib import read_csv as rc


if __name__ == "__main__":
    print("main")

    N_x = 100
    input_musk  = [0, 0, 1, 1, 0, 1, 0, 0, 1, 1] # 5
    # input_musk = None
    output_musk = [1, 1, 1, 0, 1, 0, 0, 1, 0, 1] # 6

    x = np.array([i+10 for i in range(10)])
    x_in = np.array([i+20 for i in range(10)])

    # print([x for x in output_musk if x != 0])

    input_num = N_x if input_musk == None else np.sum(input_musk)
    output_num = N_x if output_musk == None else len([x for x in output_musk if x != 0])


    # print(input_num, output_num)


    x_prime = x[np.nonzero(input_musk)]
    # print(x_prime)

    x_in *= input_musk
    # print(x_in)

    outputMusk = np.array([1 if i<32 else 0 for i in range(N_x)]) # 出力ノード数
    # print(outputMusk.shape, outputMusk, outputMusk.sum())

    periodValue = [0, 1, 0]
    periodDura = [3, 5, 2]
    # print([(periodDura[i], periodValue[i]) for i in range(len(periodValue))])
    # trainPeriod = np.concatenate(np.full((periodDura[i], periodValue[i]) for i in range(len(periodValue))))
    # print(trainPeriod)

    csv_filepath = "./input/"
    csv_filename = [
        "data_10-5_N2_300.csv", 
        "data_10-6_N2_300.csv", 
        "data_10-7_N2_300.csv", 
        "data_10-8_N2_300.csv", 
        "data_10-9_N2_300.csv", 
        "data_0_N2_300.csv"
        ]
    csvFname = []
    for fname in csv_filename:
        csvFname.append(csv_filepath + fname)
    TEST = ["10-5", "10-6"]

    
    csvData, inputData, csvDatasMean, csvDatasStd = rc.readCsvAll(csvFname, 300)
    testData = []
    if TEST is None: # テスト対象未指定なら全部でやる
        TEST = ["10-5", "10-6", "10-7", "10-8", "10-9", "0"]
    print(csvDatasMean["10-5"].reshape(-1, 1).shape)
    for test in TEST:
        # testData = np.concatenate((testData, csvDatasMean[test].reshape(-1, 1)), axis=1)
        testData.extend(csvDatasMean[test].reshape(-1, 1))
    
    testData = np.array(testData)
    print(testData.shape)

