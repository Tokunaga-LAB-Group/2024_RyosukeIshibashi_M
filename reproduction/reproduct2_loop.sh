# reproduction配下のファイル実行用シェル
# 大体指定できるようにしてるけど，込み入ったことやるなら元ファイルをいじる必要あり


# 同時実行するジョブの最大数
MAX_CONCURRENT_JOBS=5

# 実行しているジョブ数を出力する
running_jobs_count() {
  # -r は running のジョブだけを出力するオプション
  jobs -r | wc -l
}


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
FIG_SAVE_PATH="../output/20241014/"
FIG_SAVE_NAME="result_105_01.png"

for MODE in serial parallel both
do

for STIM in -5 -6 -7 -8 -9 -0
do

for RES_SEED in 1014 1024 1034 1044 1054
do

    # 実行しているジョブが最大数に達している場合は終了を待機する
    while (( $(running_jobs_count) >= MAX_CONCURRENT_JOBS )); do
        sleep 1
    done

    FIG_NAME="result_both_${STIM}_04.png"

    python3 ./reproduction21.py \
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
        --reservoir_seed ${RES_SEED} \
        --mode ${MODE} \
        --N_x 300 \
        --lamb 0.24 \
        --rho 0.9 \
        --leaking_rate 0.03 \
        --tikhonov_beta 0.00001 \
        --figure_save_path ${FIG_SAVE_PATH} \
        --figure_save_name ${FIG_NAME} \
        &\

done

done 

done

wait

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
