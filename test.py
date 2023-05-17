# なんかいろいろテスト用

import numpy as np
from tqdm import tqdm


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
    print(outputMusk.shape, outputMusk, outputMusk.sum())

