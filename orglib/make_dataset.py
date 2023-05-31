# 学習用の時系列データを作る

import numpy as np
import numpy.random as rd


# 正弦波データ生成
# クラスとして定義
class SinData():

	# コンストラクタ
	def __init__(self, n, sa, fn):
		'''
		param n: サンプリング数(この値でデータ数が決まる)
		param sa: 周波数が変化する平均周期(分散は周期*0.1ぐらい？)
		param fn: 周波数の変化のクラス数
		'''
		self.n = n
		self.secAverage = sa
		self.freqNum = fn
		# 周波数変化のクラス設定
		self.freq = []
		self.amp = 3
		for i in range(self.freqNum):
			# 周波数を適当に変化させる
			self.freq.append((i+1)*self.amp)


	# 指定長さの正弦波生成
	def makeSinData(self, s, seed):
		'''
		param s: 正弦波全体の長さ(秒)
		param seed: 乱数のシード値
		return: 周波数が変化する正弦波
		'''
		self.data = [] # 生成データ
		self.label = [] # 教師ラベル

		# データ生成
		rd.seed(seed) # シード値設定
		count = 0
		shift = 0
		# tLim = rd.normal(loc=self.secAverage, scale=self.secAverage*0.1) # 周期設定(秒)
		tLim = self.secAverage
		freq = self.freq[rd.randint(self.freqNum)] # 周波数設定
		for i in range(s*self.n):
			if(count/self.n > tLim):
				shift += freq * count / self.n
				count = 0
				# print(shift)
				# tLim = rd.normal(loc=self.secAverage, scale=self.secAverage*0.1) # 周期設定(秒)
				tLim = self.secAverage
				freq = self.freq[rd.randint(self.freqNum)] # 周波数設定
			self.data.append(np.sin(2 * np.pi * (freq * (count/self.n) + shift)))
			self.label.append(freq)
			count += 1
		
		return np.array(self.data)

	# 教師ラベル取得
	def getLabel(self):
		return np.array(self.label)
	# ラベルをクラスごとに分ける
	def getLabelArray(self):
		# ラベルをクラスごとに分ける
		result = []
		for label in self.label:
			work = np.zeros(self.freqNum, int)
			work[(label-1)//self.amp] = 1
			result.append(work)
		return np.array(result)


	# 評価指標

	# MSE
	def labelMSE(self, pred):
		diff=np.subtract(self.label,pred)
		square=np.square(diff)
		return square.mean()
	def dataMSE(self, pred):
		diff=np.subtract(self.data,pred)
		square=np.square(diff)
		return square.mean()

	# RMSE
	def labelRMSE(self, pred):
		return np.sqrt(self.labelMSE(pred))
	def dataRMSE(self, pred):
		return np.sqrt(self.dataMSE(pred))

	# 決定係数
	def labelR2(self, pred):
		labalAve = np.average(self.label)
		return 1 - np.sum((self.label - pred)**2) / np.sum((self.label - labalAve)**2)
	def dataR2(self, pred):
		dataAve = np.average(self.data)
		return 1 - np.sum((self.data - pred)**2) / np.sum((self.data - dataAve)**2)



# 正弦波データ生成その2
# 周波数は同じで位相がpiだけ異なる正弦波
class InvSinData():

	# コンストラクタ
	def __init__(self, det, f):
		'''
		param det: サンプリング数(1/2周期分)
		param f: 周波数
		'''
		self.detail = det
		self.freq = f


	# 1/2周期の正弦波を生成
	def genHalfSin(self, phase):
		'''
		param phase: 位相
		return: 指定された位相だけずれた1/2周期の正弦波
		'''
		x = np.linspace(0, 1/2, self.detail)
		return np.sin(2 * np.pi * self.freq * x + phase)

	# データ生成
	def makeSinData(self, fn, cp=10, seed=0):
		'''
		param fn: 波の数
		param cp: 変化点の数
		param seed: シード値
		return: 生成した正弦波の値
		'''
		rd.seed(seed)

		self.data = []		# 正弦波の値 
		self.label = []		# 位相(0,1)
		T = 1/self.freq
		for i in range(fn):
			if i%(fn//cp) == 0:
				if rd.random() < 0.5:
					flag = True
				else:
					flag = False
			phase = (i*self.freq) * np.pi
			if flag:
				self.label.extend(np.full(self.detail, 0))
				self.data.extend(self.genHalfSin(phase + 0))
			else:
				self.label.extend(np.full(self.detail, 1))
				self.data.extend(self.genHalfSin(phase + np.pi))

		return np.array(self.data)

	
	# ラベル取得
	def getLabel(self): # シンプル
		return np.array(self.label)
	def getLabelArray(self): # 2次元にして渡す
		result = []
		for label in self.label:
			work = np.zeros(2, int)
			work[label] = 1
			result.append(work)
		return np.array(result)
	
	# データ取得
	def getData(self): # シンプル
		return np.array(self.data)


	# 評価指標

	# MSE
	def labelMSE(self, pred):
		diff=np.subtract(self.label, pred)
		square=np.square(diff)
		return square.mean()
	def dataMSE(self, pred):
		diff=np.subtract(self.data, pred)
		square=np.square(diff)
		return square.mean()

	# RMSE
	def labelRMSE(self, pred):
		return np.sqrt(self.labelMSE(pred))
	def dataRMSE(self, pred):
		return np.sqrt(self.dataMSE(pred))

	# NRMSE
	def labelNRMSE(self, pred):
		return self.labelRMSE(pred) / np.sqrt(np.var(pred))
	def dataNRMSE(self, pred):
		return self.dataRMSE(pred) / np.sqrt(np.var(pred))


# 矩形波生成関数
def makeSquareWave(conc, dura):
	'''
	param conc: concentration, 矩形波の大きさ, リストで与える
	param dura: duration, 矩形波の長さ, リストで与える
	return: 矩形波
	'''
	data = []
	for c, d in zip(conc, dura):
		work = [c] * d
		data.extend(work)
	
	return np.array(data)
# 線虫用のデータ生成関数
def makeDiacetylData(conc, dura):
	'''
	param conc: concentration, 濃度, リストで与える
	param dura: duration, 持続時間, リストで与える
	return: ジアセチル水溶液の濃度変化をシミュレーションした値
	'''
	return makeSquareWave(conc, dura)
# データ生成の別バージョン
def makeDiacetylData2(source):
	'''
	param source: 濃度と持続時間がセットになったタプル
	return ジアセチル水溶液の濃度変化をシミュレーションした値
	'''
	conc, dura = source
	return makeDiacetylData(conc, dura)


# nakanishi made
# RNNにデータを通すために整形
def makeDataset_N(raw_data):
    data, target = [], []
    maxlen = 25
    
    for i in range(len(raw_data) - maxlen):
        data.append(raw_data[i : i + maxlen])
        target.append(raw_data[i + maxlen])
    
    re_data = np.array(data).reshape(len(data), maxlen, 1)
    re_target = np.array(target).reshape(len(data), 1)

    return re_data, re_target

# g -> 学習データ, h -> 学習ラベル
# g, h = makeDataset(f)

# 窓長のあるデータ
def makeDatasetWithWindow(raw_label, raw_data, ws):
    '''
    param raw_label: 元ラベル(1次元)
    param raw_data: 元データ(1次元)
    param ws: 窓長, window size
    return: 学習ラベル，学習データ
    '''
    data, label = [], []
    maxlen = ws
    
    for i in range(len(raw_data) - maxlen):
        label.append(raw_label[i : i + maxlen])
        data.append(raw_data[i + maxlen])
    
    re_label = np.array(label).reshape(len(label), maxlen)
    re_data = np.array(data).reshape(len(label), 1)

    return re_label, re_data
