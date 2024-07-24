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
FILENAME="data_10-5_N2_300.csv data_10-6_N2_300.csv data_10-7_N2_300.csv data_10-8_N2_300.csv data_10-9_N2_300.csv data_0_N2_300.csv"

# train時の値
TRAIN_VALUE="0 1 0"
TRAIN_DURATION="300 200 200"

# test時の値
TEST_VALUE="0 1 0"
TEST_DURATION="300 200 200"
# test時に予測するジアセチルの濃度(複数可)
TEST="10-6"

# 画像保存場所
FIG_SAVE_PATH="../output/20240627/"
FIG_SAVE_NAME="classification_106_99.png"


for FEEDBACK in 0.5 0.2 0.1 0.05 0.01
do 

for CSV_SEED in 116 126 136 146 156
do

for RES_SEED in 721 722 723 724 725
do

for LEAK_RATE in 0.1 0.3 0.5 0.7 0.9  # 0.01 0.03 0.05 0.07 0.09
do 

for BETA in 0.001 0.0001 0.00001 0.000001 0.0000001
do 

    # 実行しているジョブが最大数に達している場合は終了を待機する
    while (( $(running_jobs_count) >= MAX_CONCURRENT_JOBS )); do
        sleep 1
    done

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
        --N_x 400 \
        --lamb 0.24 \
        --rho 0.9 \
        --leaking_rate ${LEAK_RATE} \
        --tikhonov_beta ${BETA} \
        --feedback_scale ${FEEDBACK} \
        --figure_save_path ${FIG_SAVE_PATH} \
        --figure_save_name ${FIG_SAVE_NAME} \
        --csv_seed ${CSV_SEED} \
        --reservoir_seed ${RES_SEED} \
        & \


done

done

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
