import sys
import pathlib
# 実行ファイルのあるディレクトリの絶対パスを取得
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append( str(current_dir) + "/../" )

from orglib import stegano as st
import numpy as np
import re
from tqdm import tqdm
import glob


# steganoを施された画像から任意のデータを取り出す
def readData(filename, dataname):
    """
    param filename: 読み込むファイル名
    param dataname: 読み取りたいデータの名前
    retrun: 対応したデータ(文字列)
    """

    text = st.stgRead(filename)
    lines = np.array(text.split("\n"))

    data = []
    # NRMSEで始まる文字列探索
    for line in lines:
        if line.startswith(dataname):
            data = line
            break
    
    return data


# steganoを施された画像からNRMSEの値を取り出す
def readNRMSE(filename):
    """
    param filename: 読み込むファイル名
    return: NRMSEの値(float)
    """

    nrmse = readData(filename, "NRMSE")

    # print(nrmse)

    # number = float(re.search(r'\d+\.\d+', nrmse).group())

    # print(number)

    return float(re.search(r'\d+\.\d+', nrmse).group())


if __name__ == "__main__":

    filenames = [
        "/home/ishibashi/Reservoir_ESN/output/20231113/test_data_all_N2_300_result_typical_10-9_01.png",
        "/home/ishibashi/Reservoir_ESN/output/20231113/test_data_all_N2_300_result_typical_10-9_02.png",
        "/home/ishibashi/Reservoir_ESN/output/20231113/test_data_all_N2_300_result_typical_10-9_03.png",
        "/home/ishibashi/Reservoir_ESN/output/20231113/test_data_all_N2_300_result_typical_10-9_04.png",
        "/home/ishibashi/Reservoir_ESN/output/20231113/test_data_all_N2_300_result_typical_10-9_05.png"
    ]

    # 読む
    nrmses = []
    for fname in tqdm(filenames):
        nrmses.append(readNRMSE(fname))
    nrmses = np.array(nrmses)


    print(nrmses)
    print("mean = " + str(nrmses.mean()))
    print("var  = " + str(nrmses.var()))

