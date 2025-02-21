# reproduction配下のファイル実行用シェル
# 大体指定できるようにしてるけど，込み入ったことやるなら元ファイルをいじる必要あり


# ファイルパスとファイル名(複数可)
FILEPATH="../input/"
#FILENAME="data_10-5_N2_300.csv data_10-6_N2_300.csv data_10-7_N2_300.csv data_10-8_N2_300.csv data_10-9_N2_300.csv data_0_N2_300.csv"
FILENAME="data_10-6_N2_300.csv"

# train時の値
TRAIN_VALUE="0 1 0"
TRAIN_DURATION="300 200 200"

# test時の値
TEST_VALUE="0 1 0"
TEST_DURATION="300 200 200"
# test時に予測するジアセチルの濃度(複数可)
TEST="10-6"

# 画像保存場所
FIG_SAVE_PATH="../output/20240718/"
FIG_SAVE_NAME="classification_106_23.png"


python3 ./classification_03.py \
    --csv_filepath ${FILEPATH} \
    --csv_filename ${FILENAME} \
    --data_length 700 \
    --train_value ${TRAIN_VALUE} \
    --train_duration ${TRAIN_DURATION} \
    --test_value ${TEST_VALUE} \
    --test_duration ${TEST_DURATION} \
    --bias 0 \
    --test_name ${TEST} \
    --reservoir_num 1 \
    --N_x 600 \
    --lamb 0.24 \
    --rho 0.9 \
    --leaking_rate 0.1 \
    --tikhonov_beta 0.0001 \
    --figure_save_path ${FIG_SAVE_PATH} \
    --figure_save_name ${FIG_SAVE_NAME} \




# nohup python ./test.py \
#     --GPU 1 \
#     --test_path ${TEST_PATH} \
#     --save_path ${SAVE_PATH} \
#     --save_compare \
#     --img_size 256 256 --img_type 'gray' \
#     --mask_hight 32 --mask_width 32 \
#     --trained_c_weights ${W_C} \
#     --trained_d_weights ${W_D} \
#     --select_metric 'mae' \
#     > "./Logs/${SAVE_NAME}_test.log" &
