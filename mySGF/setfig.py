import matplotlib.pyplot as plt
import numpy as np

# How to use matplotlib.pyplot
# fig = plt.figure(figsize=figsize)
# ax = fig.add_subplot(
#     111,
#     xlabel=r"$\theta$", ylabel=r"$\pi$",
#     xrange=(0, 1), yrange=(0, 1),
# )
# ax.plot(x, y, label="aaa")
# ax.legend()
# fig.savefig('fig.png')

# 縦軸のラベル位置調整
# ax.set_ylabel('ylabel', labelpad=10.0) # default:4.0
# fig.align_labels()
# fig.align_labels([ax1, ax2, ax3])

Square = np.array([6.4, 6.4])
Short = np.array([9.6, 6.4])
Long = np.array([12.8, 6.4])

def init():
    params = {
        # 'mathtext.fontset':'stix',
        # 'text.usetex':True, # 日本語の時はオフ
        'font.family':['Yu Gothic', 'Hiragino sans', 'Meirio', 'Takao'],
        'font.sans-serif': 'Yu Gothic',
        'font.size':12, # 軸の数字の文字サイズ
        'axes.titlesize':20, # 図のタイトルの文字サイズ
        'axes.labelsize':18, # 軸ラベルの文字サイズ
        'legend.fontsize':15, # 凡例の文字サイズ
        'axes.linewidth':1,
        'axes.titlepad':15,
        'axes.spines.top':False,
        'axes.spines.right':False,
        'xtick.direction':'in',
        'ytick.direction':'in',
        # 'xtick.top':True,
        # 'ytick.right':True,
        'xtick.minor.visible':True,
        'ytick.minor.visible':True,
        'lines.linewidth':1.5,
        'figure.dpi':300,
        'figure.subplot.left':0.2,
        'figure.subplot.right':0.9,
        'figure.subplot.bottom':0.2,
        'figure.subplot.top':0.9,
        'figure.figsize':Short,
        'legend.fancybox':False,
        'legend.framealpha':1,
        'legend.edgecolor':'black',
        'legend.handlelength':1,
        'legend.borderaxespad':0,
        'savefig.bbox':'tight',
        'text.latex.preamble':r'\usepackage{amsmath,bm}'
    }
    plt.rcParams.update(params)

def legend(ax, place='upper right', outside=False):
    if outside:
        ax.legend(
            bbox_to_anchor=(1.01, 1),
            loc=2,
            borderaxespad=0.0
        )
    else:
        ax.legend(loc=place)