# 新しいデータがxlsx形式だったのでとりあえずcsvにする

import os
import pandas as pd

if __name__ == "__main__":
    dir_path = "/home/ishibashi/Reservoir_ESN/input/Original_Data"
    output_dir_path = "/home/ishibashi/Reservoir_ESN/input/Original_Data_csv"

    for current_dir, sub_dirs, files_list in os.walk(dir_path): 
    #     print(u"現在のディレクトリは {} です".format(current_dir)) 
    #     print(u"サブディレクトリは {} です".format(sub_dirs)) 
    #     print(u"ディレクトリ内のファイルは {} です".format(files_list)) 
    #     print("//////////////////////////////////////////////")
    
        for file_name in files_list: 
            split_dir = current_dir.split("/", 6)[-1]
            replace_file_name = file_name.replace("xlsx", "csv")
            
            # print(split_dir, file_name)
            # エクセルのシート名取得
            xlsx_path = os.path.join(current_dir, file_name)
            xlsx_file = pd.ExcelFile(xlsx_path)
            xlsx_sheets = xlsx_file.sheet_names
            # print(xlsx_sheets)

            # xlsx読み込み
            df_sheet = pd.read_excel(os.path.join(current_dir, file_name), sheet_name=xlsx_sheets[0], header=None, index_col=None)
            # print(df_sheet)

            # 作るcsvファイルの名前
            csvFileDir = os.path.join(output_dir_path, split_dir)
            csvFinePath = os.path.join(csvFileDir, replace_file_name)
            print(csvFinePath)

            # csvファイルとして保存
            # print(df_sheet.to_csv(header=False, index=False))
            #作成しようとしているディレクトリが存在するかどうかを判定する
            if os.path.isdir(csvFileDir):
                #既にディレクトリが存在する場合は何もしない
                pass
            else:
                #ディレクトリが存在しない場合のみ作成する
                os.makedirs(csvFileDir)
            
            df_sheet.to_csv(csvFinePath, header=False, index=False)
