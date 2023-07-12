# reproduction配下のファイル実行用シェル
# 大体指定できるようにしてるけど，込み入ったことやるなら元ファイルをいじる必要あり


# ファイルパスとファイル名(複数可)
FILEPATH="../input/"
FILENAME="data_10-5_N2_300.csv data_10-6_N2_300.csv data_10-7_N2_300.csv data_10-8_N2_300.csv data_10-9_N2_300.csv data_0_N2_300.csv"

# train時の値
TRAIN_VALUE="0 1 0"
TRAIN_DURATION="300 200 200"

# test時の値
TEST_VALUE="0 1 0"
TEST_DURATION="300 200 200"
# test時に予測するジアセチルの濃度(複数可)
TEST="10-5"

# 画像保存場所
FIG_SAVE_PATH="../output/"
FIG_SAVE_NAME="test_data_all_N2_300_result_01.png"


python3 ./reproduction_test.py \
    --csv_filepath ${FILEPATH} \
    --csv_filename ${FILENAME} \
    --data_length 700 \
    --train_value ${TRAIN_VALUE} \
    --train_duration ${TRAIN_DURATION} \
    --test_value ${TEST_VALUE} \
    --test_duration ${TEST_DURATION} \
    --test_name ${TEST} \
    --N_x 400 \
    --lamb 0.24 \
    --rho 0.9 \
    --leaking_rate 0.1 \
    # --figure_save_path ${FIG_SAVE_PATH}\
    # --figure_save_name ${FIG_SAVE_NAME}\


