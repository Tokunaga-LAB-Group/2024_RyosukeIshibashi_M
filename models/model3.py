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
    def __init__(self, inDim, outDim, nodeNum, lamb, rho, activationFunc, leakingRate, bias=False, seed=0):
        '''
        param inDim: 入力次元
        param outDim: 出力次元
        param nodeNum: リザバーのノード数
        param lamb: ネットワークの平均結合距離
        param rho: リカレント結合重み行列のスペクトル半径
        param activationFunc: ノードの活性化関数(式をそのまま渡す)
        param leakingRate: leaky integratorモデルのリーク率(時間スケール)
        param bias: バイアスの有無(入力値を直接出力層に持っていくか否か)
        param seed: 内部結合初期化のシード値
        '''
        super().__init__(inDim, outDim)
        self.nodeNum = nodeNum
        self.lamb = lamb
        self.rho = rho
        self.activationFunc = activationFunc
        self.leakingRate = leakingRate
        self.bias = bias
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
        print("density =" , nx.density(G))

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
    def __call__(self, inputVector, U=None):
        '''
        param inputVector: 入力状態ベクトル
        param U: バイアス用の入力データ
        return: 更新後の値(cupy)
        '''
        # ノード数と同じshapeにリサイズ(追加分は0埋め)
        inputVector = cp.resize(inputVector, self.nodeNum)
        inputVector[len(inputVector):] = 0

        self.internalState = \
        (1.0 - self.leakingRate) * self.internalState + \
        self.leakingRate * self.activationFunc(cp.dot(self.internalConnection, self.internalState) + inputVector)

        # # バイアスの設定(保留)
        # if self.bias:
        #     if U is None:
        #         ValueError("BIAS value is not exist!")
        #     self.internalState = cp.append(U, self.internalState)[:-1]

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
                  "bias":self.bias, "seed":self.seed}

        info = super().info()
        info.update(myInfo)

        return info



# 出力層
class OutputLayer(BaseLayer):
    # 出力結合重み行列の初期化
    def  __init__(self, inDim, outDim, seed=0):
        '''
        param inDim: 入力次元
        param outDim: 出力次元
        param seed: 内部結合初期化のシード値
        '''
        super().__init__(inDim, outDim)
        # 正規分布に従う乱数
        cp.random.seed(seed=seed)
        self.internalConnection = cp.random.normal(size=(outDim, inDim))


    # 学習済みの出力結合重み行列を設定
    def setOptWeight(self, incnOpt):
        '''
        param incnOpt: 学習済みの出力結合重み
        '''
        super().setIntCon(incnOpt)



# # 出力フィードバック
# class Feedback:
#     # フィードバック結合重み行列の初期化
#     def __init__(self, N_y, N_x, fb_scale, seed=0):
#         '''
#         param N_y: 出力次元
#         param N_x: リザバーのノード数
#         param fb_scale: フィードバックスケーリング(フィードバックの強さ)
#         '''
#         # 一様分布に従う乱数
#         np.random.seed(seed = seed)
#         self.Wfb = np.random.uniform(-fb_scale, fb_scale, (N_x, N_y))
#         # print(self.Wfb.shape)
    
#     # フィードバック結合重み行列による重みづけ
#     def __call__(self, y):
#         '''
#         param y: N_y次元のベクトル
#         return: N_x次元のベクトル
#         '''
#         # print(np.dot(self.Wfb, y).shape)
#         return cp.asnumpy(cp.dot(self.Wfb, y))



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
    
        # 各ハイパーパラメータの情報
    def info(self):
        '''
        return: クラスメンバの名称と値
        '''
        info = {"method":"Tikhonov", "nodeNum":self.nodeNum, "outputDimention":self.outputDimention, "beta": self.beta}

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
                 noiseLevel = None,
                 outputFunc = identify,
                 invOutputFunc = identify,
                 classification = False,
                 averageWindow = None,
                ):
        '''
        param noiseLevel: 入力に付与するノイズの大きさ
        param outputFunc: 出力層の非線形関数
        param invOutputFunc: output_funcの逆関数←何に使うの...?
        param classification: 分類問題の場合はtrue
        param averageWindow: 分類問題で平均出力する窓幅
        '''
        self.inputLayer = inputLayer
        self.reservoirLayer = reservoirLayer
        self.outputLayer = outputLayer
        self.y_prev = cp.zeros(reservoirLayer.nodeNum)
        self.outputFunc = outputFunc
        self.invOutputFunc = invOutputFunc
        self.classification = classification
        self.params = {"InputLayer":self.inputLayer.info(), 
                       "ReservoirLayer":self.reservoirLayer.info(),
                       "OutputLayer":self.outputLayer.info(),
                       "ESN":{"outputFunc":outputFunc, "invOutputFunc":invOutputFunc, 
                              "classification":classification, "averageWindow":averageWindow}}

        # # 出力層からリザバーへのフィードバックの有無
        # if fb_scale is None:
        #     self.Feedback = None
        # else:
        #     self.Feedback = Feedback(N_y, N_x, fb_scale, fb_seed)

        # リザバーの状態更新におけるノイズの有無
        if noiseLevel is None:
            self.noise = None
        else:
            cp.random.seed(seed=0)
            self.noise = cp.random.uniform(-noiseLevel, noiseLevel, (self.reservoirLayer.nodeNum))

        # 分類問題か否か
        if classification:
            if averageWindow is None:
                raise ValueError("Window for time average is not given!")
            else:
                self.window = cp.zeros((averageWindow, self.reservoirLayer.nodeNum))


    # バッチ学習
    def train(self, U, D, optimizer:Tikhonov, transLen = None):
        '''
        param U: 入力データ，データ長*inputDimention
        param D: 入力データに対する正解データ，データ長*outputDimention
        param optimizer: 学習器
        param transLen: 過渡期の長さ
        return: 学習前のモデル出力，データ長*outputDimention
        '''
        trainLen = len(U)
        if transLen is None:
            transLen = 0 # デフォルトで0にすればいいのでは？
        Y = cp.empty(0)

        # 時間発展
        for n in tqdm(range(trainLen)):

            #### input layer
            inputVector = self.inputLayer(U[n])

            # # フィードバック結合
            # if self.Feedback is not None:
            #     x_back = self.Feedback(self.y_prev)
            #     # x_back[self.input_num:] = 0
            #     # print(x_in.shape, x_back.shape)
            #     x_in += x_back
            
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
            outputVector = self.outputLayer(reservoirVector)
            Y = cp.append(Y, self.outputFunc(outputVector))
            # self.prevOutputVector = grandTruth # フィードバックで使う

        # 学習済みの出力結合重み行列を設定
        self.outputLayer.setOptWeight(optimizer.getWoutOpt())

        # モデル出力
        return Y
    

    # バッチ学習後の予測
    def predict(self, U):
        '''
        param U: 入力データ，データ長*inputDimention
        return: 学習後のモデル出力
        '''
        testLen = len(U)
        predictY = cp.empty(0)

        # 時間発展
        for n in range(testLen):

            #### input layer
            inputVector = self.inputLayer(U[n])

            # # フィードバック結合
            # if self.Feedback is not None:
            #     # print(self.y_prev.shape)
            #     x_back = self.Feedback(self.y_prev)
            #     # x_back[self.input_num:] = 0
            #     x_in += x_back
            
            #### Reservoir layer
            reservoirVector = self.reservoirLayer(inputVector)

            # 分類問題の場合は窓幅分の平均を取得
            if self.classification:
                self.window = cp.append(self.window, reservoirVector.reshape(1, -1), axis = 0)
                self.window = cp.delete(self.window, 0, 0)
                reservoirVector = cp.average(self.window, axis = 0)

            #### output layer

            # 学習後のモデル出力
            outputVector = self.outputLayer(reservoirVector)
            predictY = cp.append(predictY, self.outputFunc(outputVector))
            # self.y_prev = y_pred

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

        return json.dumps(self.params, ensure_ascii=False, indent=4)
    
    def infoCSV(self):
        '''
        retrun 一部パラメータのcsv形式データ
        '''
        resData = self.params["ReservoirLayer"]
        data = [resData['nodeNum'], resData['lamb'], resData['rho'], resData['leakingRate']]
        return data



