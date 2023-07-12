# csvからヒートマップ作成


import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb


# csvファイルからデータを読み込む
def readParamFreq(filename):

	# ファイル開く
	F = open(filename, "r")

	#ファイルからデータを読み込み
	rows = csv.reader(F)
	
	# for文で行を1つずつ取り出す
	data = []
	for row in rows: 
		data.append(row)
		print(row) # rowの中身を表示

	# ファイル閉じる
	F.close()

	return data



if __name__ == "__main__":
    # データ読み込み

    path = "../output/"
    
    # df_flights = sb.load_dataset('flights')
    # print(df_flights)
    # data = readParamFreq("output/check_param_freq0.5_2.csv")
    data = pd.read_csv(path + "csv_data/voice_03.csv")
    print(data.head(5)) 
    
    # 加工
    data_pivot = pd.pivot_table(data=data, values="NRMSE", columns="leaking_rate", index="tikhonov_beta", aggfunc=np.mean)    
    
    # 表示
    # sb.heatmap(data_pivot)
    plt.figure(figsize=(15, 8))
    sb.heatmap(data_pivot, annot=True, fmt='g', cmap='Blues')
    plt.savefig(path + "/nrmse_heat_03.png")
    # plt.show()