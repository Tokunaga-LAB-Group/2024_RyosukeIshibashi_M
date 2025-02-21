# ESNのモデルを設定
# 基本はReservoir_C-Elegans_Qに準じる



# 必要ライブラリ読み込み
import sys
import pathlib
import json
import cupy as cp
import networkx as nx
from tqdm import tqdm

# 実行ファイルのあるディレクトリの絶対パスを取得
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append( str(current_dir) + "\..\\" )
from orglib import graph_maker as gm



######## ここから関数定義 #########

# 恒等写像
def identify(x):
    return x


#### BaseLayer定義(全てのLayerはこのクラスを継承する) ####

class BaseLayer:
    """
    input: 状態ベクトル
    output: 何らかの処理を行った後の状態ベクトル
    """

    def __init__(self, inDim, outDim):
        '''
        param inDim: 入力次元
        param outDim: 出力次元
        '''
        self.inputDimention = inDim
        self.outputDimention = outDim

        # 内部結合設定は各クラスで個別にやる
        self.internalConnection = cp.empty(0) # バグるかも
        self.internalState = cp.empty(0)

    def setIntCon(self, incn):
        '''
        param incn: 内部結合行列
        '''
        self.internalConnection = incn

    # 入力結合重み行列による重みづけ
    def __call__(self, inputVector):
        '''
        param inputVector: 入力状態ベクトル
        return: 更新後の値(cupy)
        '''
        self.internalState = inputVector

        return cp.dot(self.internalConnection, self.internalState)
    
    # 各ハイパーパラメータの情報
    def info(self):
        '''
        return: クラスメンバの名称と値
        '''
        info = {"inputDimention": self.inputDimention, "outputDimention":self.outputDimention}
        
        return info



#### リザバー本体の実装 ####

# 入力層
class InputLayer(BaseLayer):
    # 入力結合重み行列W_inの初期化
    def __init__(self, inDim, outDim, inputScale, seed=0):
        '''
        param inDim: 入力次元
        param outDim: 出力次元
        param inputScale: 入力スケーリング
        param seed: 内部結合初期化のシード値
        '''
        super().__init__(inDim, outDim)
        self.inputScale = inputScale
        self.seed = seed
        # 一様分布に従う乱数
        cp.random.seed(seed=seed)

        # 内部結合設定
        self.internalConnection = cp.random.uniform(-inputScale, inputScale, (outDim, inDim))

    # 各ハイパーパラメータの情報
    def info(self):
        '''
        return: クラスメンバの名称と値
        '''
        myInfo = {"inputScale":self.inputScale, "seed":self.seed}

        info = super().info()
        info.update(myInfo)

        return info




# リザバー
class ReservoirLayer(BaseLayer):
    # リカレント結合重み行列Wの初期化
    def __init__(self, inDim, outDim, nodeNum, lamb, rho, activationFunc, leakingRate, seed=0):
        '''
        param inDim: 入力次元
        param outDim: 出力次元
        param nodeNum: リザバーのノード数
        param lamb: ネットワークの平均結合距離
        param rho: リカレント結合重み行列のスペクトル半径
        param activationFunc: ノードの活性化関数(式をそのまま渡す)
        param leakingRate: leaky integratorモデルのリーク率(時間スケール)
        param seed: 内部結合初期化のシード値
        '''
        super().__init__(inDim, outDim)
        self.nodeNum = nodeNum
        self.lamb = lamb
        self.rho = rho
        self.activationFunc = activationFunc
        self.leakingRate = leakingRate
        self.seed = seed
        self.internalConnection = self.make_connection(nodeNum, lamb, rho) # リカレント結合重み行列の生成
        self.internalState = cp.zeros(nodeNum) # リザバー状態ベクトルの初期化


    # リカレント結合重み行列の生成
    def make_connection(self, nodeNum, lamb, rho):
        '''
        param nodeNum: リザバーのノード数
        param lamb: ネットワークの平均結合距離
        param rho: リカレント結合重み行列のスペクトル半径
        return: 初期化された内部結合
        '''
        # 距離を考慮したリザバー結合
        G = gm.makeGraph(nodeNum, lamb, self.seed)
        # print("density =" , nx.density(G))

        # 行列への変換(結合構造のみ)
        connection = nx.to_numpy_array(G)
        W = cp.array(connection)

        # 非ゼロ要素を一様分布に従う乱数として生成
        recScale = 1.0
        cp.random.seed(seed = self.seed*self.seed)
        W *= cp.random.uniform(-recScale, recScale, (nodeNum, nodeNum))

        # スペクトル半径の計算
        eigvList, _ = cp.linalg.eigh(W)
        spRadius = cp.max(cp.abs(eigvList))

        # 指定のスペクトル半径rhoに合わせてスケーリング
        W *= rho / spRadius

        return W
    
    # リザバー状態ベクトルの更新
    def __call__(self, inputVector, dropout=None):
        '''
        param inputVector: 入力状態ベクトル
        param dropout: リザバー層のドロップアウト指定
        return: 更新後の値(cupy)
        '''
        # ノード数と同じshapeにリサイズ
        inputVector = cp.resize(inputVector, self.nodeNum)
        # inputVector[len(inputVector):] = 0

        self.internalState = \
        (1.0 - self.leakingRate) * self.internalState + \
        self.leakingRate * self.activationFunc(cp.dot(self.internalConnection, self.internalState) + inputVector)

        return self.internalState[:self.outputDimention]
    
    # リザバー状態ベクトルの初期化
    def resetReservoirState(self):
        self.internalState = cp.zeros(self.nodeNum)

    # 各ハイパーパラメータの情報
    def info(self):
        '''
        return: クラスメンバの名称と値
        '''
        myInfo = {"nodeNum":self.nodeNum, "lambda":self.lamb, "rho":self.rho,
                  "activationFunc":self.activationFunc, "leakingRate":self.leakingRate,
                  "seed":self.seed}

        info = super().info()
        info.update(myInfo)

        return info


# マルチリザバー(パラレルとシリアルをまとめたクラス)
class MultiReservoirLayer(BaseLayer):
        # 入力結合重み行列W_inの初期化
    def __init__(self, inDim, outDim, layers:list[tuple[InputLayer, ReservoirLayer]], mode, intensity):
        '''
        param inDim: 入力次元
        param outDim: 出力次元
        param layers: (入力層, リザバー層) のリスト (入力層の入力次元はinDimと合わせる)
        param mode: リザバー層の並べ方とか parallel, serial, both
        param intensity: リザバー層間のデータ受け渡し倍率
        '''
        super().__init__(inDim, outDim)
        self.layers = layers
        self.mode = mode
        # intensityがリストor数値で場合分け, self.intensityはリスト固定
        self.intensity = [intensity]*len(layers) if type(intensity) != list else intensity

        # 内部結合設定
        self.internalConnection = cp.random.uniform(-1, 1, (outDim, inDim))


    # 入力結合重み行列による重みづけ
    def __call__(self, inputVector, dropout=None):
        '''
        param inputVector: 入力状態ベクトル
        param dropout: リザバー層のドロップアウト指定
        return: 更新後の値(cupy)
        '''
        self.internalState = cp.empty(0)
        if dropout == None:
            dropout=[1]*len(self.layers)
        maskValue = 0

        # エラー処理はいったん放置
        if self.mode == "parallel":
            for i in range(0, len(self.layers)):
                (input, reservoir) = self.layers[i]
                if dropout[i] == 0:
                    self.internalState = cp.append(self.internalState, cp.array([maskValue]*reservoir.outputDimention))
                    continue
                inputsVector = input(inputVector)
                reservoirsVector = reservoir(inputsVector)
                self.internalState = cp.append(self.internalState, reservoirsVector)

        elif self.mode == "serial":
            (_, reservoir) = self.layers[0]
            if dropout[0] == 0:
                # reservoirsVector = cp.array([maskValue]*reservoir.outputDimention)
                (_, reservoir) = self.layers[1]
                reservoirsVector = reservoir(inputVector)
            elif dropout[1] == 0:
                reservoirsVector = reservoir(inputVector)
                (_, reservoir) = self.layers[1]
                reservoirsVector = reservoir([maskValue]*reservoir.outputDimention)
            else:
                reservoirsVector = reservoir(inputVector)
                (_, reservoir) = self.layers[1]
                reservoirsVector = reservoir(reservoirsVector * self.intensity[1])

            for i in range(2, len(self.layers)):
                (_, reservoir) = self.layers[i]
                if dropout[i] == 0:
                    continue
                reservoirsVector = reservoir(reservoirsVector * self.intensity[i])
            self.internalState = reservoirsVector
        
        elif self.mode == "both":
            prevReservoirsVector = cp.zeros(1)
            for i in range(0, len(self.layers)):
                (input, reservoir) = self.layers[i]
                if dropout[i] == 0:
                    self.internalState = cp.append(self.internalState, cp.array([maskValue]*reservoir.outputDimention))
                    continue
                inputsVector = input(inputVector)
                reservoirsVector = reservoir(cp.append(inputsVector, prevReservoirsVector * self.intensity[i]))
                self.internalState = cp.append(self.internalState, reservoirsVector)
                prevReservoirsVector = reservoirsVector

        elif self.mode == "mixed":
            (_, reservoir) = self.layers[0]
            if dropout[0] == 0:
                reservoirsVector = cp.array([maskValue]*reservoir.outputDimention)
                self.internalState = reservoirsVector
                (_, reservoir) = self.layers[1]
                reservoirsVector = reservoir(inputVector)
                self.internalState = cp.append(self.internalState, reservoirsVector)
            elif dropout[1] == 0:
                reservoirsVector = reservoir(inputVector)
                self.internalState = reservoirsVector
                (_, reservoir) = self.layers[1]
                reservoirsVector = reservoir([maskValue]*reservoir.outputDimention)
                self.internalState = cp.append(self.internalState, reservoirsVector)
            else:
                reservoirsVector = reservoir(inputVector)
                self.internalState = reservoirsVector
                (_, reservoir) = self.layers[1]
                reservoirsVector = reservoir(reservoirsVector * self.intensity[1])
                self.internalState = cp.append(self.internalState, reservoirsVector)
            
            for i in range(2, len(self.layers)):
                (_, reservoir) = self.layers[i]
                if dropout[i] == 0:
                    self.internalState = cp.append(self.internalState, cp.array([maskValue]*reservoir.outputDimention))
                    continue
                reservoirsVector = reservoir(reservoirsVector * self.intensity[i])
                self.internalState = cp.append(self.internalState, reservoirsVector)


        return self.internalState


    # リザバー状態ベクトルの初期化
    def resetReservoirState(self):
        for (_, reservoir) in self.layers:
            reservoir.resetReservoirState()


    # 各ハイパーパラメータの情報
    def info(self):
        '''
        return: クラスメンバの名称と値
        '''
        layersInfo = {}
        for i in range(len(self.layers)):
            (input, reservoir) = self.layers[i]
            layersInfo[f"layer[{i}]"] = {"inputsLayer":input.info() if not input==None else None, "reservoirLayer":reservoir.info()}
        myInfo = {"mode":self.mode, "layers":layersInfo, "intensity":self.intensity}

        info = super().info()
        info.update(myInfo)

        return info


# パラレルリザバー
class ParallelReservoirLayer(MultiReservoirLayer):
    # 初期化
    def __init__(self, inDim, outDim, layers:list[tuple[InputLayer, ReservoirLayer]]):
        '''
        param inDim: 入力次元
        param outDim: 出力次元 (リザバー層の出力次元の合計と合わせる)
        param layers: (入力層, リザバー層) のリスト (入力層の入力次元はinDimと合わせる)
        '''
        super().__init__(inDim, outDim, layers, "parallel", 0)

# シリアルリザバー
class SerialReservoirLayer(MultiReservoirLayer):
    # 初期化
    def __init__(self, inDim, outDim, layers:list[ReservoirLayer], intensity):
        '''
        param inDim: 入力次元
        param outDim: 出力次元 (リストの最後のリザバー層の出力次元と合わせる)
        param layers: リザバー層のリスト (あるリザバー層の出力次元とその直後のリザバー層の入力次元を合わせる)
        param intensity: リザバー層間のデータ受け渡し倍率
        '''
        layersMulti = []
        for res in layers:
            layersMulti.append((None, res))
        super().__init__(inDim, outDim, layersMulti, "serial", intensity)

# 複合リザバー
class BothReservoirLayer(MultiReservoirLayer):
    # 初期化
    def __init__(self, inDim, outDim, layers:list[tuple[InputLayer, ReservoirLayer]], intensity):
        '''
        param inDim: 入力次元
        param outDim: 出力次元 (リザバー層の出力次元の合計と合わせる)
        param layers: (入力層, リザバー層) のリスト (入力層の入力次元はinDimと合わせる)
        param intensity: リザバー層間のデータ受け渡し倍率
        '''
        super().__init__(inDim, outDim, layers, "both", intensity)

# 複合リザバーその2
class MixedReservoirLayer(MultiReservoirLayer):
    # 初期化
    def __init__(self, inDim, outDim, layers:list[ReservoirLayer], intensity):
        '''
        param inDim: 入力次元
        param outDim: 出力次元 (リストの最後のリザバー層の出力次元と合わせる)
        param layers: リザバー層のリスト (あるリザバー層の出力次元とその直後のリザバー層の入力次元を合わせる)
        param intensity: リザバー層間のデータ受け渡し倍率
        '''
        layersMulti = []
        for res in layers:
            layersMulti.append((None, res))
        super().__init__(inDim, outDim, layersMulti, "mixed", intensity)


# 出力層
class OutputLayer(BaseLayer):
    # 出力結合重み行列の初期化
    def  __init__(self, inDim, outDim, bias=False, ud=0, seed=0):
        '''
        param inDim: 入力次元
        param outDim: 出力次元
        param bias: バイアスの有無(入力値を直接出力層に持っていくか否か)
        param ud: 入力値の次元
        param seed: 内部結合初期化のシード値
        '''
        super().__init__(inDim, outDim)
        
        # バイアスの設定
        self.bias = bias
        if self.bias:
            self.inputDimention += ud

        # 正規分布に従う乱数
        cp.random.seed(seed=seed)
        self.internalConnection = cp.random.normal(size=(self.outputDimention, self.inputDimention))

    # バイアスに対応
    def __call__(self, inputVector, U):
        '''
        param inputVector: 入力状態ベクトル
        param U: モデルへの入力値
        return: 更新後の値(cupy)
        '''
        if self.bias:
            inputVector = cp.append(inputVector, U)

        return super().__call__(inputVector)

    # 学習済みの出力結合重み行列を設定
    def setOptWeight(self, incnOpt):
        '''
        param incnOpt: 学習済みの出力結合重み
        '''
        super().setIntCon(incnOpt)

    # 各ハイパーパラメータの情報
    def info(self):
        '''
        return: クラスメンバの名称と値
        '''
        info = {"inputDimention": self.inputDimention, "outputDimention":self.outputDimention, "bias":self.bias}
        
        return info



# 出力フィードバック
class FeedbackLayer(BaseLayer):
    # フィードバック結合重み行列の初期化
    def __init__(self, inDim, outDim, feedbackScale, seed=0):
        '''
        param inDim: 入力次元 (出力層の出力次元と揃える)
        param outDim: 出力次元 (リザバー層の入力次元と揃える)
        param fb_scale: フィードバックスケーリング(フィードバックの強さ)
        param seed: 内部結合初期化のシード値
        '''
        super().__init__(inDim, outDim)
        self.feedbackScale = feedbackScale
        self.seed = seed

        # 一様分布に従う乱数
        cp.random.seed(seed = seed)
        # 内部結合設定
        self.internalConnection = cp.random.uniform(-feedbackScale, feedbackScale, (outDim, inDim))
        # print(self.Wfb.shape)
    

    # 各ハイパーパラメータの情報
    def info(self):
        '''
        return: クラスメンバの名称と値
        '''
        myInfo = {"feedbackScale":self.feedbackScale, "seed":self.seed}

        info = super().info()
        info.update(myInfo)

        return info



#### リザバー本体の実装はここまで ####


#### ここから回帰用の関数 ####


# # Moore-Penrose疑似逆行列
# class Pseudoinv:
#     def __init__(self, N_x, N_y):
#         '''
#         param N_x: リザバーのノード数
#         param N_y: 出力次元
#         '''
#         self.X = np.empty((N_x, 0))
#         self.D = np.empty((N_y, 0))
    
#     # 状態集積行列及び教師集積行列の更新
#     def __call__(self, d, x):
#         x = np.reshape(x, (-1, 1))
#         d = np.reshape(d, (-1, 1))
#         self.X = np.hstack((self.X, x))
#         self.D = np.hstack((self.D, d))
    
#     # Woutの最適解(近似解)の導出
#     def get_Wout_opt(self):
#         Wout_opt = cp.asnumpy(cp.dot(self.D, cp.linalg.pinv(self.X)))
#         return Wout_opt


# リッジ回帰(beta=0の時は線形回帰)
class Tikhonov:
    def __init__(self, outLayerInDim, outLayerOutDim, beta):
        '''
        param outLayerInDim: 出力層の入力次元
        param outLayerOutDim: 出力層の出力次元
        param beta: 正則化パラメータ
        '''
        self.outLayerInDim = outLayerInDim
        self.outLayerOutDim = outLayerOutDim
        self.beta = beta
        self.X_XT = cp.zeros((outLayerInDim, outLayerInDim))
        self.D_XT = cp.zeros((outLayerOutDim, outLayerInDim))
    
    # 学習用の行列の更新
    def __call__(self, d, x):
        '''
        param d: 目標値
        param x: リザバー層の内部状態(出力層の入力次元に合わせる)
        '''
        x = cp.reshape(x, (-1, 1))
        d = cp.reshape(d, (-1, 1))
        self.X_XT += cp.dot(x, x.T)
        self.D_XT += cp.dot(d, x.T)
    
    # Woutの最適解(近似解)の導出
    def getWoutOpt(self):
        XpseudoInv = cp.linalg.inv(self.X_XT + self.beta * cp.identity(self.outLayerInDim))
        WoutOpt = cp.dot(self.D_XT, XpseudoInv)

        return WoutOpt
    
    # 初期化
    def resetValue(self):
        self.X_XT = cp.zeros((self.outLayerInDim, self.outLayerInDim))
        self.D_XT = cp.zeros((self.outLayerOutDim, self.outLayerInDim))

    # 各ハイパーパラメータの情報
    def info(self):
        '''
        return: クラスメンバの名称と値
        '''
        info = {"method":"Tikhonov", "outLayerInDim":self.outLayerInDim, "outLayerOutDim":self.outLayerOutDim, "beta": self.beta}

        return info



# # 逐次最小二乗法(RLS法)
# class RLS:
#     def __init__(self, N_x, N_y, delta, lam, update):
#         '''
#         pamam N_x: リザバーのノード数
#         param N_y: 出力次元
#         param delta: 行列Pの初期条件の係数(P = delta * I, 0 < delta < 1)
#         param lam: 忘却係数 (0 < lam < 1, 1に近い値)
#         param update: 各時刻での更新繰り返し回数
#         '''
#         self.delta = delta
#         self.lam = lam
#         self.update = update
#         self.P = (1.0 / self.delta) * np.eye(N_x, N_x)
#         self.Wout = np.zeros([N_y, N_x])
    
#     # Woutの更新←なにしてるか全然わからん
#     def __call__(self, d, x):
#         x = np.reshape(x, (-1, 1))
#         for i in np.arange(self.update):
#             v = d - cp.dot(self.Wout, x)
#             gain = (1 / self.lam * cp.dot(self.P, x))
#             gain = gain / (1 + 1 / self.lam * cp.dot(cp.dot(x.T, self.P), x))
#             self.P = 1 / self.lam * (self.P - cp.dot(cp.dot(gain, x.T), self.P))
#             self.Wout += cp.asnumpy(cp.dot(v, gain.T))

#         return self.Wout


#### 回帰用の関数の実装ここまで ####

######## 関数定義ここまで ########



#######################################
#                                     #
#    ここからなんかモデル実装になる？    #
#                                     #
#######################################


# エコーステートネットワーク(ESN)
# 各層は外部で定義する
# このクラスで提供するのは学習とその結果を得る機能
class ESN: 
    # 各層の初期化
    def __init__(self,
                 inputLayer: InputLayer,
                 reservoirLayer: ReservoirLayer,
                 outputLayer: OutputLayer,
                 feedbackLeyer:FeedbackLayer = None,
                 outputFunc = identify,
                 invOutputFunc = identify,
                 noiseLevel = None,
                 classification = False,
                 averageWindow = None,
                ):
        '''
        param inputLayer: 入力層
        param reservoirLayer: リザバー層
        param outputLayer: 出力層
        param feedbackLayer: フィードバック層 (teacher forcing 専用)
        param outputFunc: 出力層の非線形関数
        param invOutputFunc: output_funcの逆関数←何に使うの...?
        param noiseLevel: 入力に付与するノイズの大きさ
        param classification: 分類問題の場合はtrue
        param averageWindow: 分類問題で平均出力する窓幅
        '''
        self.inputLayer = inputLayer
        self.reservoirLayer = reservoirLayer
        self.outputLayer = outputLayer
        self.feedbackLayer = feedbackLeyer
        self.outputFunc = outputFunc
        self.invOutputFunc = invOutputFunc
        self.noiseLevel = noiseLevel
        self.classification = classification
        self.params = {"InputLayer":self.inputLayer.info(), 
                       "ReservoirLayer":self.reservoirLayer.info(),
                       "OutputLayer":self.outputLayer.info(),
                       "FeedbackLayer":self.feedbackLayer.info() if not self.feedbackLayer == None else None,
                       "ESN":{"outputFunc":outputFunc, "invOutputFunc":invOutputFunc, 
                              "noiseLevel":noiseLevel,
                              "classification":classification, "averageWindow":averageWindow}}

        # 出力層からリザバー層へのフィードバック用のベクトル
        self.prevOutputVector = cp.zeros(self.outputLayer.outputDimention)[0] # エラー出ると思う

        # リザバーの状態更新におけるノイズの有無
        if noiseLevel is None:
            self.noise = None
        else:
            cp.random.seed(seed=0)
            self.noise = cp.random.uniform(-noiseLevel, noiseLevel, (self.reservoirLayer.inputDimention))

        # 分類問題か否か
        if classification:
            if averageWindow is None:
                raise ValueError("Window for time average is not given!")
            else:
                self.window = cp.zeros((averageWindow, self.reservoirLayer.outputDimention))


    # バッチ学習
    def train(self, U, D, optimizer:Tikhonov, transLen = None):
        '''
        param U: 入力データ，データ長*inputDimention
        param D: 入力データに対する正解データ，データ長*outputDimention
        param optimizer: 学習器
        param transLen: 過渡期の長さ
        return: 学習前のモデル出力，データ長*outputDimention
        '''

        # 最適化手法についての情報記録
        self.params["optimizer"] = optimizer.info()

        trainLen = len(U)
        if transLen is None:
            transLen = 0 # デフォルトで0にすればいいのでは？
        Y = cp.empty((0, self.outputLayer.outputDimention))

        # 時間発展
        for n in tqdm(range(trainLen)):

            #### input layer
            inputVector = self.inputLayer(U[n])

            # フィードバック結合
            if self.feedbackLayer is not None:
                feedbackVector = self.feedbackLayer(self.prevOutputVector)
                inputVector += feedbackVector
            
            # ノイズ付与
            if self.noise is not None:
                inputVector += self.noise
            

            #### Reservoir layer
            reservoirVector = self.reservoirLayer(inputVector)

            # 分類問題の場合は窓幅分の平均を取得(要修正)
            if self.classification:
                self.window = cp.append(self.window, reservoirVector.reshape(1, -1), axis = 0)
                self.window = cp.delete(self.window, 0, 0)
                reservoirVector = cp.average(self.window, axis = 0)
            
            #### output layer

            # 目標値
            grandTruth = D[n]
            grandTruth = self.invOutputFunc(grandTruth)

            # 学習器
            if n > transLen: # 過渡期を過ぎたら
                optimizer(grandTruth, self.reservoirLayer.internalState[:self.outputLayer.inputDimention])

            # 学習前のモデル出力
            outputVector = self.outputLayer(reservoirVector, U[n])
            Y = cp.vstack((Y, self.outputFunc(outputVector)))
            self.prevOutputVector = grandTruth # フィードバックで使う

        # 学習済みの出力結合重み行列を設定
        self.outputLayer.setOptWeight(optimizer.getWoutOpt())

        # モデル出力
        return Y
    

    # ミニバッチ学習
    def trainMini(self, U, D, optimizer:Tikhonov, transLen=None, dropout=None, changePoint=0):
        '''
        param U: 入力データ，ミニバッチ数*inputDimention*データ長
        param D: 入力データに対する正解データ，ミニバッチ数*outputDimention*データ長
        param optimizer: 学習器
        param transLen: 過渡期の長さ
        param dropout: リザバー層のドロップアウトを設定 初期値None listの0/1で指定
        param changePoint: マスクを行う切り替えポイント(ポイント以降でマスク)
        return: 学習前のモデル出力， outputDimention*データ長
        '''

        # 最適化手法についての情報記録
        self.params["optimizer"] = optimizer.info()

        WoutOpt = cp.empty(0)
        Y = cp.empty((0, self.outputLayer.outputDimention))
        miniSize = 0
        
        for udi in tqdm(range(len(U))):
            # ミニバッチごとにデータ取り出し
            u = U[udi]
            d = D[udi]
            miniSize += 1

            trainLen = len(u)
            if transLen is None:
                transLen = 0 # デフォルトで0にすればいいのでは？

            # 時間発展
            for n in range(trainLen):

                #### input layer
                inputVector = self.inputLayer(u[n])

                # フィードバック結合
                if self.feedbackLayer is not None:
                    feedbackVector = self.feedbackLayer(self.prevOutputVector)
                    inputVector += feedbackVector

                # ノイズ付与
                if self.noise is not None:
                    inputVector += self.noise


                #### Reservoir layer
                reservoirVector = self.reservoirLayer(inputVector, None if udi<changePoint else dropout)

                # 分類問題の場合は窓幅分の平均を取得(要修正)
                if self.classification:
                    self.window = cp.append(self.window, reservoirVector.reshape(1, -1), axis = 0)
                    self.window = cp.delete(self.window, 0, 0)
                    reservoirVector = cp.average(self.window, axis = 0)

                #### output layer

                # 目標値
                grandTruth = d[n]
                grandTruth = self.invOutputFunc(grandTruth)

                # 学習器
                if n > transLen: # 過渡期を過ぎたら
                    optimizer(grandTruth, self.reservoirLayer.internalState[:self.outputLayer.inputDimention])

                # 学習前のモデル出力
                outputVector = self.outputLayer(reservoirVector, u[n])
                Y = cp.vstack((Y, self.outputFunc(outputVector)))
                self.prevOutputVector = grandTruth # フィードバックで使う

            # # ミニバッチ単位で重み計算
            # if len(WoutOpt) == 0:
            #     WoutOpt = optimizer.getWoutOpt()
            # WoutOpt += optimizer.getWoutOpt()

            # # 重み初期化
            # optimizer.resetValue()
            # リザバー層の内部状態初期化
            self.reservoirLayer.resetReservoirState()

        # 学習済みの出力結合重み行列を設定
        # self.outputLayer.setOptWeight(WoutOpt / miniSize)
        self.outputLayer.setOptWeight(optimizer.getWoutOpt())

        # モデル出力
        return Y
    

    # バッチ学習後の予測
    def predict(self, U, dropout=None):
        '''
        param U: 入力データ，データ長*inputDimention
        param dropout: リザバー層のドロップアウトを設定 初期値None listの0/1で指定
        return: 学習後のモデル出力
        '''
        testLen = len(U)
        predictY = cp.empty((0, self.outputLayer.outputDimention))

        # 時間発展
        for n in range(testLen):

            #### input layer
            inputVector = self.inputLayer(U[n])

            # フィードバック結合
            if self.feedbackLayer is not None:
                feedbackVector = self.feedbackLayer(self.prevOutputVector)
                inputVector += feedbackVector

            #### Reservoir layer
            reservoirVector = self.reservoirLayer(inputVector, dropout)

            # 分類問題の場合は窓幅分の平均を取得
            if self.classification:
                self.window = cp.append(self.window, reservoirVector.reshape(1, -1), axis = 0)
                self.window = cp.delete(self.window, 0, 0)
                reservoirVector = cp.average(self.window, axis = 0)

            #### output layer

            # 学習後のモデル出力
            outputVector = self.outputLayer(reservoirVector, U[n])
            predictY = cp.vstack((predictY, self.outputFunc(outputVector)))
            self.prevOutputVector = outputVector # エラー出ると思う

        # モデル出力(学習後)
        return predictY

    # # バッチ学習後の予測(自律系のフリーラン)
    # def run(self, U):
    #     test_len = len(U)
    #     Y_pred = []
    #     y = U[0]

    #     # 時間発展
    #     for n in range(test_len):
    #         x_in = self.Input(y)

    #         # フィードバック結合
    #         if self.Feedback is not None:
    #             x_back = self.Feedback(self.y_prev)
    #             x_in += x_back

    #         # リザバー状態ベクトル
    #         x = self.Reservoir(x_in)

    #         # 学習後のモデル出力
    #         y_pred = self.Output(x)
    #         Y_pred.append(self.output_func(y_pred))
    #         y = y_pred
    #         self.y_prev = y
        
    #     return np.array(Y_pred)


    # # オンライン学習と予測
    # def adapt(self, U, D, optimizer):
    #     '''
    #     param U: 教師データの入力，データ長*N_u
    #     param D: 教師データの出力，データ長*N_y
    #     param optimizer: 学習器
    #     return: よくわかんない
    #     '''
    #     data_len = len(U)
    #     Y_pred = []
    #     Wout_abs_mean = []

    #     # 出力結合重み更新
    #     for n in np.arange(0, data_len, 1):
    #         x_in = self.Input(U[n])
    #         x = self.Reservoir(x_in)
    #         d = D[n]
    #         d = self.inv_output_func(d)

    #         # 学習
    #         Wout = optimizer(d, x)

    #         # モデル出力
    #         y = np.dot(Wout, x)
    #         Y_pred.append(y)
    #         Wout_abs_mean.append(np.mean(np.abs(Wout)))
        
    #     return np.array(Y_pred), np.array(Wout_abs_mean)

    def info(self):
        '''
        return: 各種パラメータの値の文字列
        '''
        # text = ""
        # for category, info in self.params.items():
        #     text += f"{category} : {{ \n"
        #     for key, value in info.items():
        #         text += f"{key} : {value}\n"
        #     text += "}\n"

        # objectをJSONにするためのやつ
        def func(obj):
            try:
                return obj.__dict__
            except AttributeError:
                return f"{obj}"
            

        return json.dumps(self.params, ensure_ascii=False, indent=4, default=func)
    
    def infoCSV(self):
        '''
        retrun 一部パラメータのcsv形式データ
        '''
        resData = self.params["ReservoirLayer"]
        data = [resData['nodeNum'], resData['lamb'], resData['rho'], resData['leakingRate']]
        return data



