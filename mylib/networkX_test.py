# networkXについて色々調査

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm



# ノード間の距離を求める
def dist(a, b):
    '''
    param a: 距離を求めたいノード 型はdict
    param b: 距離を求めたいノードその2 型はdict
    return: ノード間の距離
    '''
    return  np.linalg.norm(a-b)

# トーラス物体上の距離を求める
# 端っこをなくしたい
def distTorus(a, b, range=2.0):
    '''
    param a: 距離を求めたいノード 型はdict
    param b: 距離を求めたいノードその2 型はdict
    param range: 座標の最大値(正方形のみ想定)
    retrun: トーラス物体上のノード間の距離
    '''
    # 各軸の距離の絶対値を計算
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])

    # 各軸の距離の絶対値がrangeの半分を越えたらrange分引く
    if(dx >= range/2):
        dx -= range
    if(dy >= range/2):
        dy -= range

    return math.sqrt(dx**2 + dy**2)




# 極座標変換する
def convertCircle(pos):
    '''
    直交座標を無理やり極座標にする(xをtheta, yをrとする)
    param pos: 直交座標で表された座標
    retrun: 極座標で表された座標
    '''
    retPos = {}
    # 変換
    for i, [theta, r] in pos.items():
        # print(i, theta, r)
        rad = 2 * math.pi * theta
        x = math.sqrt(r) * math.cos(rad)
        y = math.sqrt(r) * math.sin(rad)
        retPos[i] = np.array([x, y], dtype=np.float32)
    
    # print(retPos)
    return retPos


# ノード間に接続を作ったり作らなかったり
def connect(G, a, b, pos, lam, selfLoop=0.05, c=1.0, seed=0):
    '''
    param G: 接続を行うグラフ
    param a: 接続を生みたいノードのkey 型はint
    param b: 接続を生みたいノードのkeyその2 型はint
    param pos: 各ノードの位置を記録した行列
    param lam: 平均接続距離
    param selfLoop: 自己ループ結合確率
    param c: 接続率を操作する定数
    param seed: 乱数のシード値
    return: 接続を増やしたりしたグラフG
    '''
    C = c # ある距離内でどのくらい接続するか設定する定数

    # 接続確率計算
    p = C * math.exp(-distTorus(pos[a], pos[b], 1)**2 / lam**2)

    # 接続判定
    # np.random.seed(seed)
    if a == b : # 自己ループ結合を抑制
        if selfLoop > np.random.random_sample():
            nx.add_path(G, [a, b])
    else:
        if p > np.random.random_sample():
            nx.add_path(G, [a, b])

    return G



def makeGraph(N_x, lamb, seed=0):
    '''
    param N_x: ノード数
    param lamb: 平均接続距離
    param seed: 乱数のシード値
    '''

    # 空のグラフ生成
    G = nx.empty_graph(N_x, nx.DiGraph)

    # レイアウトの取得
    pos = nx.random_layout(G, seed=seed)
    # pos = nx.rescale_layout_dict(pos, scale=1.0) # 最大値を2にする
    # pos = convertCircle(pos)
    # print(pos)

    # distAll = []
    # 接続する
    for i in tqdm(range(N_x)):
        for j in range(N_x):
            connect(G, i, j, pos, lamb)
            # lambda:0.240 ~ density:0.05
            # lambda:0.350 ~ density:0.10
            # lambda:0.440 ~ density:0.15
            # distAll.append(distTorus(pos[i], pos[j], 1))

    # distAll = np.array(distAll)
    # print(distAll.shape, distAll.mean(), distAll.max(), distAll.min())

    # nx.draw(G, pos, node_size=20, arrowsize=3)
    # plt.show()

    return G




if __name__ == "__main__":

    sm = 16
    sn = 16

    # 2次元グリッドグラフ
    G2d = nx.grid_2d_graph(sm, sn)

    # ランダムグラフ
    N_x = 400
    density = 0.05
    m = int(N_x * (N_x - 1) * density / 2) # 総結合数
    # Gr = nx.gnm_random_graph(N_x, 0, 0)

    # print(G2d)

    # 結合状況
    connection = nx.to_numpy_matrix(G2d)
    
    # print(connection.shape)
    # print(connection)

    # レイアウトの取得
    # pos = nx.random_layout(Gr, seed=101)
    # print(pos)

    # リスケール←要らない
    # pos_r = nx.rescale_layout_dict(pos, scale=math.sqrt(N_x/10)/2)

    # distAll = []
    # 接続する
    Gr = makeGraph(N_x, 0.1)
    # distAll = np.array(distAll)


    # 可視化
    # pos = nx.spring_layout(Gr)
    # print(pos[0])
    # nx.draw(Gr, pos, node_size=50)
    # plt.show()

    print(Gr)
    print(nx.density(Gr))
    # print(distAll.shape, distAll.mean(), distAll.max(), distAll.min())
    
    
    # G = nx.Graph()
    # nx.add_star(G, [0, 1, 2, 3])
    # nx.add_star(G, [10, 11, 12], weight=2)
    # nx.add_path(G, [0, 1])
    # nx.add_path(G, [1, 0])
    # nx.add_path(G, [10, 11, 12], weight=7)
    # nx.add_cycle(G, [0, 1, 2, 3])
    # nx.add_cycle(G, [10, 11, 12], weight=7)

    # pos = nx.kamada_kawai_layout(G)
    # nx.draw(G, pos)
    # plt.show()

    # print(pos[0], G[11], nx.density(Gr))

    # print(dist(pos_r[0], pos_r[1]))

    # print(G, G[0])
    # print(nx.to_numpy_matrix(G))
