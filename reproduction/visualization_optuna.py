
import optuna
import matplotlib.pyplot as plt

# You can use Matplotlib instead of Plotly for visualization by simply replacing `optuna.visualization` with
# `optuna.visualization.matplotlib` in the following examples.
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_rank
from optuna.visualization.matplotlib import plot_slice
from optuna.visualization.matplotlib import plot_timeline



if __name__ == "__main__" :

    # 結果のロード
    study = optuna.load_study(
        study_name="reproduct_sample07", 
        storage="mysql://ishibashi@127.0.0.1/ishibashi_optuna_01",
    )

    # 可視化

    # fig = plot_contour(study)
    # fig = plot_edf(study)
    # fig = plot_intermediate_values(study)
    # fig = plot_optimization_history(study)
    # fig = plot_parallel_coordinate(study)
    # fig = plot_param_importances(study)
    fig = plot_rank(study)
    # fig = plot_slice(study)
    # fig = plot_timeline(study)


    # 表示のカスタマイズ
    # print(fig)
    # fig.set_ylim(0, 1)
    # plt.ylim(0.4, 2)


    fname = "/home/ishibashi/Reservoir_ESN/output/20241101/" + "optuna_rank03.png"

    # 保存
    plt.savefig(fname, bbox_inches="tight", pad_inches=0.05, dpi=400)

    plt.show()
    plt.close()

    # 最適解出力
    best_params = study.best_params
    found_nodeNum = best_params["nodeNum"]
    found_leaking_rate = best_params["leaking_rate"]
    found_tikhonov_beta = best_params["tikhonov_beta"]
    
    best_value = study.best_value

    print(f"Found nodeNum: {found_nodeNum}, Found leaking rate:{found_leaking_rate}, Found tikhonov beta : {found_tikhonov_beta}, value : {best_value}")

