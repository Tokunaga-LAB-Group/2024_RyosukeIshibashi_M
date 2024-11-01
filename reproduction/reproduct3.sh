# reproduction配下のファイル実行用シェル
# 大体指定できるようにしてるけど，込み入ったことやるなら元ファイルをいじる必要あり


# ファイルパスとファイル名(複数可)
FILEPATH="../input/"
FILENAME="data_unveiled_fig5_ab.json"

# train時の値
TRAIN_VALUE="0 1 0"
TRAIN_DURATION="300 200 200"

# test時の値
TEST_VALUE="0 1 0"
TEST_DURATION="300 200 200"
# test時に予測するジアセチルの濃度(複数可)
TEST="10-5"
STIMULATE=-5

# 画像保存場所
FIG_SAVE_PATH="../output/20241101/"
FIG_SAVE_NAME="result_105_01.png"


for STIM in -6 #-5 -6 -7 -8 -9 -0
do

FIG_NAME="result_both_${STIM}_01.png"

python3 ./reproduction22.py \
    --json_filepath ${FILEPATH} \
    --json_filename ${FILENAME} \
    --stimulate ${STIM} \
    --data_length 600 \
    --train_value ${TRAIN_VALUE} \
    --train_duration ${TRAIN_DURATION} \
    --test_value ${TEST_VALUE} \
    --test_duration ${TEST_DURATION} \
    --bias 0.1 \
    --test_name ${TEST} \
    --reservoir_num 1 \
    --reservoir_seed 12235 \
    --mode both \
    --N_x 300 \
    --lamb 0.24 \
    --rho 0.9 \
    --leaking_rate 0.03 \
    --tikhonov_beta 0.00001 \
    --figure_save_path ${FIG_SAVE_PATH} \
    --figure_save_name ${FIG_NAME} \

done


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
