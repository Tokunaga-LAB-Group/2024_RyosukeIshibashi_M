# ESNのモデルを設定
# 基本はReservoir_C-Elegans_Qに準じる



# 必要ライブラリ読み込み
import sys
import pathlib
# 実行ファイルのあるディレクトリの絶対パスを取得
current_dir = pathlib.Path(__file__).resolve().parent
# モジュールのあるパスを追加
sys.path.append( str(current_dir) + "\..\\" )
import numpy as np
import networkx as nx
import math
from tqdm import tqdm
from orglib import graph_maker as gm



######## ここから関数定義 #########

# 恒等写像
def identify(x):
    return x


#### リザバー本体の実装 ####

# 入力層
class Input:
    # 入力結合重み行列W_inの初期化
    def __init__(self, N_u, N_x, input_scale, seed=0):
        '''
        param N_u: 入力次元
        param N_x: リザバーのノード数
        param input_scale: 入力スケーリング
        '''
        # 一様分布に従う乱数
        np.random.seed(seed=seed)
        self.Win = np.random.uniform(-input_scale, input_scale, (N_x, N_u))

        # 入力ノード設定
        # print(self.Win.shape) # デバッグ用
        # self.Win[input_num:] = 0
        # print(self.Win.shape)
    
    # 入力結合重み行列Winによる重みづけ
    def __call__(self, u):
        '''
        param u: N_u次元のベクトル
        return: N_x次元のベクトル
        '''
        # print(self.Win.shape, u.shape)
        return np.dot(self.Win, u)



# リザバー
class Reservoir:
    # リカレント結合重み行列Wの初期化
    def __init__(self, N_x, lamb, rho, activation_func, leaking_rate, seed=0):
        '''
        param N_x: リザバーのノード数
        param density: ネットワークの結合密度
        param rho: リカレント結合重み行列のスペクトル半径
        param activation_func: ノードの活性化関数(式をそのまま渡す)
        param leaking_rate: leaky integratorモデルのリーク率(時間スケール)
        '''
        self.seed = seed
        self.W = self.make_connection(N_x, lamb, rho) # リカレント結合重み行列の生成
        self.x = np.zeros(N_x) # リザバー状態ベクトルの初期化
        self.activation_func = activation_func
        self.alpha = leaking_rate


    # リカレント結合重み行列の生成
    def make_connection(self, N_x, lamb, rho):
        # 距離を考慮したリザバー結合
        G = gm.makeGraph(N_x, lamb, self.seed)
        print("density =" , nx.density(G))

        # 行列への変換(結合構造のみ)
        connection = nx.to_numpy_matrix(G)
        W = np.array(connection)

        # 非ゼロ要素を一様分布に従う乱数として生成
        rec_scale = 1.0
        np.random.seed(seed = self.seed*self.seed)
        W *= np.random.uniform(-rec_scale, rec_scale, (N_x, N_x))

        # スペクトル半径の計算
        eigv_list = np.linalg.eig(W)[0]
        sp_radius = np.max(np.abs(eigv_list))

        # 指定のスペクトル半径rhoに合わせてスケーリング
        W *= rho / sp_radius

        return W
    
    # リザバー状態ベクトルの更新
    def __call__(self, x_in):
        '''
        param x_in: リザバーへの入力層からの入力
        return: 更新後の状態ベクトル
        '''
        # self.x = self.x.reshape(-1, 1)
        self.x = (1.0 - self.alpha) * self.x + self.alpha * self.activation_func(np.dot(self.W, self.x) + x_in)

        return self.x
    
    # リザバー状態ベクトルの初期化
    def reset_reservoir_state(self):
        self.x *= 0.0


# 出力層
class Output:
    # 出力結合重み行列の初期化
    def  __init__(self, N_x, N_y, output_num, seed=0):
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param output_num: 出力ノード数
        '''
        # 正規分布に従う乱数
        np.random.seed(seed=seed)
        self.Wout = np.random.normal(size=(N_y, output_num))

        # 出力ノード設定
        # self.input_num = input_num
        # self.output_num = output_num
        # print(self.Wout.shape, np.count_nonzero(self.Wout == 0)) # デバッグ用
        # self.Wout[:, :self.input_num] = 0
        # self.Wout[:, self.input_num + self.output_num:] = 0
        # print(self.Wout.shape, np.count_nonzero(self.Wout == 0))

    # 出力結合重み行列による重みづけ
    def __call__(self, x):
        '''
        param x: output_num次元のベクトル
        retrun: N_y次元のベクトル
        '''
        # x[:self.input_num] = 0
        # x[self.input_num + self.output_num:] = 0
        # print(x.shape, np.count_nonzero(x == 0))
        # print(np.dot(self.Wout, x).shape)
        return np.dot(self.Wout, x)

    # 学習済みの出力結合重み行列を設定
    def setweight(self, Wout_opt):
        # print(self.Wout.shape, Wout_opt.shape)
        self.Wout = Wout_opt

        # 出力ノード設定
        # print(self.Wout.shape, np.count_nonzero(self.Wout == 0)) # デバッグ用
        # self.Wout[:, :self.input_num] = 0
        # self.Wout[:, self.input_num + self.output_num:] = 0
        # print(self.Wout.shape, np.count_nonzero(self.Wout == 0))



# 出力フィードバック
class Feedback:
    # フィードバック結合重み行列の初期化
    def __init__(self, N_y, N_x, fb_scale, seed=0):
        '''
        param N_y: 出力次元
        param N_x: リザバーのノード数
        param fb_scale: フィードバックスケーリング(フィードバックの強さ)
        '''
        # 一様分布に従う乱数
        np.random.seed(seed = seed)
        self.Wfb = np.random.uniform(-fb_scale, fb_scale, (N_x, N_y))
        # print(self.Wfb.shape)
    
    # フィードバック結合重み行列による重みづけ
    def __call__(self, y):
        '''
        param y: N_y次元のベクトル
        return: N_x次元のベクトル
        '''
        # print(np.dot(self.Wfb, y).shape)
        return np.dot(self.Wfb, y)



#### リザバー本体の実装はここまで ####


#### ここから回帰用の関数 ####


# Moore-Penrose疑似逆行列
class Pseudoinv:
    def __init__(self, N_x, N_y):
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        '''
        self.X = np.empty((N_x, 0))
        self.D = np.empty((N_y, 0))
    
    # 状態集積行列及び教師集積行列の更新
    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X = np.hstack((self.X, x))
        self.D = np.hstack((self.D, d))
    
    # Woutの最適解(近似解)の導出
    def get_Wout_opt(self):
        Wout_opt = np.dot(self.D, np.linalg.pinv(self.X))
        return Wout_opt


# リッジ回帰(beta=0の時は線形回帰)
class Tikhonov:
    def __init__(self, N_x, N_y, output_mask, beta):
        '''
        param N_x: リザバーのノード数
        param N_y: 出力次元
        param output_num: 出力ノード数
        param beta: 正則化パラメータ
        '''
        self.beta = beta
        if output_mask != None:
            output_num = np.sum(output_mask)
        else:
            output_num = N_x
        self.X_XT = np.zeros((output_num, output_num))
        self.D_XT = np.zeros((N_y, output_num))
        self.N_x = N_x
        self.output_num = output_num
    
    # 学習用の行列の更新
    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.X_XT += np.dot(x, x.T)
        self.D_XT += np.dot(d, x.T)
    
    # Woutの最適解(近似解)の導出
    def get_Wout_opt(self):
        X_pseudo_inv = np.linalg.inv(self.X_XT + self.beta * np.identity(self.output_num))
        Wout_opt = np.dot(self.D_XT, X_pseudo_inv)

        return Wout_opt


# 逐次最小二乗法(RLS法)
class RLS:
    def __init__(self, N_x, N_y, delta, lam, update):
        '''
        pamam N_x: リザバーのノード数
        param N_y: 出力次元
        param delta: 行列Pの初期条件の係数(P = delta * I, 0 < delta < 1)
        param lam: 忘却係数 (0 < lam < 1, 1に近い値)
        param update: 各時刻での更新繰り返し回数
        '''
        self.delta = delta
        self.lam = lam
        self.update = update
        self.P = (1.0 / self.delta) * np.eye(N_x, N_x)
        self.Wout = np.zeros([N_y, N_x])
    
    # Woutの更新←なにしてるか全然わからん
    def __call__(self, d, x):
        x = np.reshape(x, (-1, 1))
        for i in np.arange(self.update):
            v = d - np.dot(self.Wout, x)
            gain = (1 / self.lam * np.dot(self.P, x))
            gain = gain / (1 + 1 / self.lam * np.dot(np.dot(x.T, self.P), x))
            self.P = 1 / self.lam * (self.P - np.dot(np.dot(gain, x.T), self.P))
            self.Wout += np.dot(v, gain.T)

        return self.Wout


#### 回帰用の関数の実装ここまで ####

######## 関数定義ここまで ########



#######################################
#                                     #
#    ここからなんかモデル実装になる？    #
#                                     #
#######################################


# エコーステートネットワーク(ESN)
class ESN: 
    # 各層の初期化
    def __init__(self, N_u, N_y, N_x,
                lamb = 0.135, # lambda:0.135 ~ density:0.05
                input_scale = 1.0, 
                rho = 0.95, 
                activation_func = np.tanh,
                fb_scale = None, fb_seed = 0, 
                noise_level = None,
                leaking_rate = 1.0,
                output_func = identify,
                inv_output_func = identify,
                classification = False,
                average_window = None,
                input_mask = None,
                output_mask = None
                ):
        '''
        param N_u: 入力次元
        param N_y: 出力次元
        param N_x: リザバーのノード数
        param density: リザバーのネットワーク結合密度
        param input_scale: 入力スケーリング
        param rho: リカレント結合重み行列のスペクトル半径
        param activation_func: リザバーノードの活性化関数関数
        param fb_scale: フィードバックスケーリング
        param fb_seed: フィードバック結合重み行列に使う乱数のシード値
        param leaking_rate: leaky integratorモデルのリーク率
        param output_func: 出力層の非線形関数
        param inv_output_func: output_funcの逆関数←何に使うの...?
        param classification: 分類問題の場合はtrue
        param average_window: 分類問題で平均出力する窓幅
        param input_num: 入力ノード数
        param output_num: 出力ノード数
        '''
        self.Input = Input(N_u, N_x, input_scale)
        self.Reservoir = Reservoir(N_x, lamb, rho, activation_func, leaking_rate)
        # 要素が0以外の数を与える
        input_num = N_x if input_mask == None else np.sum(input_mask)
        output_num = N_x if output_mask == None else np.sum(output_mask)
        self.Output = Output(N_x, N_y, output_num)
        self.N_u = N_u
        self.N_y = N_y
        self.N_x = N_x
        self.y_prev = np.zeros(N_y)
        self.output_func = output_func
        self.inv_output_func = inv_output_func
        self.classification = classification
        self.input_mask = input_mask
        self.output_mask = output_mask
        self.params = {"N_u":N_u, "N_y":N_y, "N_x":N_x, "lamb":lamb, "input_scale":input_scale, 
                        "rho":rho, "activation_func":activation_func, 
                        "fb_scale":fb_scale, "fb_seed":fb_seed, "leaking_rate":leaking_rate, 
                        "output_func":output_func, "inv_output_func":inv_output_func, 
                        "classification":classification, "average_window":average_window,
                        "input_mask":input_mask, "output_mask":output_mask}

        # 出力層からリザバーへのフィードバックの有無
        if fb_scale is None:
            self.Feedback = None
        else:
            self.Feedback = Feedback(N_y, N_x, fb_scale, fb_seed)

        # リザバーの状態更新におけるノイズの有無
        if noise_level is None:
            self.noise = None
        else:
            np.random.seed(seed=0)
            self.noise = np.random.uniform(-noise_level, noise_level, (self.N_x))

        # 分類問題か否か
        if classification:
            if average_window is None:
                raise ValueError("Window for time average is not given!")
            else:
                self.window = np.zeros((average_window, N_x))


    # バッチ学習
    def train(self, U, D, optimizer, trans_len = None, period = None):
        '''
        param U: 教師データの入力，データ長*N_u
        param D: 教師データの出力，データ長*N_y
        param optimizer: 学習器
        param trans_len: 過渡期の長さ
        param period: 学習区間(0,1のリストで与える)
        return: 学習前のモデル出力，データ長*N_y
        '''
        train_len = len(U)
        if trans_len is None:
            trans_len = 0 # デフォルトで0にすればいいのでは？
        if period is None:
            period = [1] * train_len
        Y = []

        # 時間発展
        for n in tqdm(range(train_len)):
            x_in = self.Input(U[n])

            # フィードバック結合
            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                # x_back[self.input_num:] = 0
                # print(x_in.shape, x_back.shape)
                x_in += x_back
            
            # ノイズ
            if self.noise is not None:
                # print(x_in.shape, self.noise.shape)
                # self.noise[self.input_num:] = 0
                x_in += self.noise
            

            # 絞込みをマスク行列で行うように変更
            # 入力の絞り込み
            if self.input_mask != None:
                x_in *= self.input_mask
            # リザバー状態ベクトル
            x = self.Reservoir(x_in)
            # 出力の絞り込み
            if self.output_mask != None:
                x_prime = x[np.nonzero(self.output_mask)]
            else:
                x_prime = x

            # 分類問題の場合は窓幅分の平均を取得
            if self.classification:
                self.window = np.append(self.window, x_prime.reshape(1, -1), axis = 0)
                self.window = np.delete(self.window, 0, 0)
                x_prime = np.average(self.window, axis = 0)
            
            # 目標値
            d = D[n]
            d = self.inv_output_func(d)

            # 学習器
            # 学習前の値にマスクを掛けておく
            if n > trans_len: # 過渡期を過ぎたら
                if period[n] == 1: # 学習可能区間なら
                    optimizer(d, x_prime)
            
            # 学習前のモデル出力
            y = self.Output(x_prime)
            Y.append(self.output_func(y))
            self.y_prev = d
        
        # 学習済みの出力結合重み行列を設定
        self.Output.setweight(optimizer.get_Wout_opt())

        # モデル出力
        return np.array(Y)

    # バッチ学習後の予測
    def predict(self, U):
        test_len = len(U)
        Y_pred = []

        # 時間発展
        for n in range(test_len):
            x_in = self.Input(U[n])

            # フィードバック結合
            if self.Feedback is not None:
                # print(self.y_prev.shape)
                x_back = self.Feedback(self.y_prev)
                # x_back[self.input_num:] = 0
                x_in += x_back
            
            # 入力の絞り込み
            if self.input_mask != None:
                x_in *= self.input_mask
            # リザバー状態ベクトル
            x = self.Reservoir(x_in)
            # 出力の絞り込み
            if self.output_mask != None:
                x_prime = x[np.nonzero(self.output_mask)]
            else:
                x_prime = x

            # 分類問題の場合は窓幅分の平均を取得
            if self.classification:
                self.window = np.append(self.window, x_prime.reshape(1, -1), axis = 0)
                self.window = np.delete(self.window, 0, 0)
                x_prime = np.average(self.window, axis = 0)
            
            # 学習後のモデル出力
            # print(x.shape)
            y_pred = self.Output(x_prime)
            Y_pred.append(self.output_func(y_pred))
            self.y_prev = y_pred

        # モデル出力(学習後)
        return np.array(Y_pred)

    # バッチ学習後の予測(自律系のフリーラン)
    def run(self, U):
        test_len = len(U)
        Y_pred = []
        y = U[0]

        # 時間発展
        for n in range(test_len):
            x_in = self.Input(y)

            # フィードバック結合
            if self.Feedback is not None:
                x_back = self.Feedback(self.y_prev)
                x_in += x_back

            # リザバー状態ベクトル
            x = self.Reservoir(x_in)

            # 学習後のモデル出力
            y_pred = self.Output(x)
            Y_pred.append(self.output_func(y_pred))
            y = y_pred
            self.y_prev = y
        
        return np.array(Y_pred)


    # オンライン学習と予測
    def adapt(self, U, D, optimizer):
        '''
        param U: 教師データの入力，データ長*N_u
        param D: 教師データの出力，データ長*N_y
        param optimizer: 学習器
        return: よくわかんない
        '''
        data_len = len(U)
        Y_pred = []
        Wout_abs_mean = []

        # 出力結合重み更新
        for n in np.arange(0, data_len, 1):
            x_in = self.Input(U[n])
            x = self.Reservoir(x_in)
            d = D[n]
            d = self.inv_output_func(d)

            # 学習
            Wout = optimizer(d, x)

            # モデル出力
            y = np.dot(Wout, x)
            Y_pred.append(y)
            Wout_abs_mean.append(np.mean(np.abs(Wout)))
        
        return np.array(Y_pred), np.array(Wout_abs_mean)

    def info(self):
        '''
        return: 各種パラメータの値の文字列
        '''
        text = ""
        for key, value in self.params.items():
            text += f"{key} = {value}\n"
        return text
    def infoCSV(self):
        '''
        retrun 一部パラメータのcsv形式データ
        '''
        data = [self.params['N_x'], self.params['lamb'], self.params['rho'], self.params['leaking_rate']]
        return data




