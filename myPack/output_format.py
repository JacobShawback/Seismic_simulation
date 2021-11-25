import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


class Format:
    def params():
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = ["ヒラギノ丸ゴ ProN W4, 16"]
        plt.rcParams['font.family'] = 'Times New Roman'  # font familyの設定
        plt.rcParams['mathtext.fontset'] = 'stix'  # math fontの設定
        plt.rcParams["font.size"] = 10  # 全体のフォントサイズが変更されます。
        plt.rcParams['xtick.labelsize'] = 8  # 軸だけ変更されます。
        plt.rcParams['ytick.labelsize'] = 8  # 軸だけ変更されます
        plt.rcParams['xtick.direction'] = 'in'  # x axis in
        plt.rcParams['ytick.direction'] = 'in'  # y axis in
        plt.rcParams['axes.labelpad'] = 6 # y axis in
        # plt.rcParams['axes.grid'] = True # make grid
        plt.rcParams["legend.loc"] = 'upper right'
        plt.rcParams["legend.fancybox"] = False  # 丸角
        plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
        plt.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更
        plt.rcParams["legend.handlelength"] = 1  # 凡例の線の長さを調節
        plt.rcParams["legend.labelspacing"] = 1.  # 垂直（縦）方向の距離の各凡例の距離
        plt.rcParams["legend.handletextpad"] = 1.  # 凡例の線と文字の距離の長さ
        plt.rcParams["legend.markerscale"] = 2  # 点がある場合のmarker scale
        plt.rcParams["legend.borderaxespad"] = 0.  # 凡例の端とグラフの端を合わせる
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['figure.figsize'] = (4.8, 3)  # figure size in inch, 横×縦
        plt.rcParams['figure.subplot.bottom'] = 0.15
        plt.rcParams['figure.subplot.left'] = 0.18
        # plt.rcParams['font.family'] ='sans-serif'#使用するフォント
        # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['xtick.direction'] = 'in'
        # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.major.width'] = 1.0  # x軸主目盛り線の線幅
        plt.rcParams['ytick.major.width'] = 1.0  # y軸主目盛り線の線幅
        # plt.rcParams['font.size'] = 11 #フォントの大きさ
        plt.rcParams['axes.linewidth'] = 1.0  # 軸の線幅edge linewidth。囲みの太さ
        plt.rcParams['lines.linewidth'] = 0.5
        return
