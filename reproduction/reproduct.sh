# reproduction配下のファイル実行用シェル
# 大体指定できるようにしてるけど，込み入ったことやるなら元ファイルをいじる必要あり


# ファイルパスとファイル名(複数可)
FILEPATH="../input/voice/"
# FILENAME="02_tsumugi_hello.wav 03_metan_hello.wav 04_meimei_hello.wav 05_aoyama_hello.wav 06_kurono_hello.wav"
FILENAME="02_tsumugi_hello.wav"

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
FIG_SAVE_NAME="voice_result_12.png"


python3 ./reproduction_voice.py \
    --csv_filepath ${FILEPATH} \
    --csv_filename ${FILENAME} \
    --data_length 700 \
    --train_value ${TRAIN_VALUE} \
    --train_duration ${TRAIN_DURATION} \
    --test_value ${TEST_VALUE} \
    --test_duration ${TEST_DURATION} \
    --test_name ${TEST} \
    --N_x 1000 \
    --lamb 0.24 \
    --rho 0.99 \
    --leaking_rate 0.9 \
    --tikhonov_beta 0.0001 \
    --noise_level 0.001 \
    --figure_save_name ${FIG_SAVE_NAME}\
    --figure_save_path ${FIG_SAVE_PATH}\
    # --feedback_scale 0.1 \


# for leak in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1
# do 

# for beta in 0.5 0.3 0.1 0.01 0.001 0.0001
# do 


# python3 ./reproduction_voice.py \
#     --csv_filepath ${FILEPATH} \
#     --csv_filename ${FILENAME} \
#     --data_length 700 \
#     --train_value ${TRAIN_VALUE} \
#     --train_duration ${TRAIN_DURATION} \
#     --test_value ${TEST_VALUE} \
#     --test_duration ${TEST_DURATION} \
#     --test_name ${TEST} \
#     --N_x 1000 \
#     --lamb 0.24 \
#     --rho 0.99 \
#     --leaking_rate ${leak} \
#     --tikhonov_beta ${beta} \
#     --noise_level 0.001 \
#     --figure_save_name ${FIG_SAVE_NAME}\
#     # --figure_save_path ${FIG_SAVE_PATH}\
#     # --feedback_scale 0.1 \


# done

# done



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
