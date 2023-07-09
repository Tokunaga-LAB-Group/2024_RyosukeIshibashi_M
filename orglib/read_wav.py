# wavファイルからデータを読み込む
# [0, 1]のfloat
# 前処理とか？
# これ要らないな

import numpy as np
import soundfile as sf
from tqdm import tqdm


# wavファイル読み込むやつ
def readWav(fname):
    '''
    param fname: 読み込むwavファイル名 拡張子は"wav"にする
    return: wavデータを配列にしたもの(numpyに変換), サンプルレート
    '''

    # モノラルとステレオでちょっと違う
    # 今回はモノラルだと思う
    # fname = '1980s-Casio-Celesta-C5.wav' # mono
    # fname = 'Alesis-Fusion-Pizzicato-Strings-C4.wav' # stereo

    data, samplerate = sf.read(fname)
    # sf.write('new_file.wav', data, samplerate)

    # print(data.shape)

    # stereo音源なら
    # l_channel = data[:,0]
    # r_channel = data[:,1]

    return np.array(data), samplerate


# 複数指定して読み込めるように
def readWavAll(fnames):
    '''
    param fnames: ファイル名．リストで複数指定可能 
    return: データを配列にしたやつを結合した奴
    '''
    datas = []
    for fname in fnames:
        d, _ = sf.read(fname)
        datas.extend(d)

    return np.array(datas)


# 訓練データと正解データを同時に読み込む
# 長さもそろえる
def readWavDataGT(trainFname, testFname):
    '''
    param trainFname: 訓練データのファイル名．配列で複数指定可能
    param tsetFname: テストデータのファイル名．一つだけ
    return: 訓練データ, 正解データ (長さを揃えた)
    '''
    trainDatas = []
    testDatas = []

    testData, _ = sf.read(testFname)
    for fname in trainFname:
        data, _ = sf.read(fname)

        # 長いほうに揃える　短いほうは0パディング
        diff = len(data) - len(testData)
        if diff > 0:
            trainDatas.extend(data)
            testDatas.extend(np.pad(testData, (0, diff)))
        else:
            trainDatas.extend(np.pad(data, (0, -diff)))
            testDatas.extend(testData)

    return np.array(trainDatas), np.array(testDatas)



# wavファイル書きこむやつ
def writeWav(fname, data, samplerate):
    '''
    param fname: 生成するファイル名 wavまで含む
    param data: 元データ．[0, 1]のfloatのlist
    param samplerate: サンプリングレート大体44k
    '''

    # 書きこみ
    sf.write(fname, data, samplerate)
