# from myPack.response_method import Response
from myPack.modal_analysis_method import ModalAnalysis
from myPack.data_processing import Data
from myPack.make_matrix import House
import numpy as np
# import matplotlib.pyplot as plt
# from scipy import sqrt, power, exp, log10
# from scipy.fftpack import fft, ifft, fftfreq
# import tarfile
# import re
# import os
# import glob
# import pandas as pd
# from datetime import datetime
from time import time
# import numpy.linalg as LA
from myPack.output_format import Format
from myPack.response_method import Response
Format.params()

# path = os.path.dirname(os.path.abspath(__file__))
# # print(path)
# os.chdir(path)

tstart = time()

# STRUCTURE PARAMETER --------------------------------------
Area_f = 71.626  # [m2]
m1 = 9760.54  # 2F weight [kg]
m2 = 6439.62  # 1F weight [kg]
mf = 29198.65  # base weight [kg]
Ifx = 570.870/Area_f*mf  # moment of inertia [kg*m2]
Ify = 343.980/Area_f*mf
k1x = 6079.97e3  # [N/m]
k2x = 6439.10e3
k1y = 6219.83e3
k2y = 6353.06e3
l1,l2 = 5.8,3.0  # [m]
kh = 2.83e9  # [N/m]
kth = 5.75e10  # [N*m/rad]
cf = 4.38e6  # [N/m]
cth = 3.69e7
house_x = House(m1,m2,mf,Ifx,k1x,k2x,kh,kth,l1,l2,cf=cf,cth=cth)
house_y = House(m1,m2,mf,Ify,k1y,k2y,kh,kth,l1,l2,cf=cf,cth=cth)

np.savetxt('data/MCK/mx_mat.txt',house_x.M)
np.savetxt('data/MCK/my_mat.txt',house_y.M)
np.savetxt('data/MCK/cx_mat.txt',house_x.C)
np.savetxt('data/MCK/cy_mat.txt',house_y.C)
np.savetxt('data/MCK/kx_mat.txt',house_x.K)
np.savetxt('data/MCK/ky_mat.txt',house_y.K)


# INPUT DATA --------------------------------------
data = Data("data/JR_Takatori_NS.acc",fname='ground/takatori')
data.Process()
data.Output()
# data.ResponseSpectrum()


# # MODAL ANALYSIS --------------------------------------
# Mode = ModalAnalysis(house_x.M, house_x.K, house_x.F)
# Mode.NaturalValue('4DOF')
# # Mode.Participation()
# # print(np.sum(Mode.beta))


# # FREQ ANALYSIS -------------------------------------
modelx = Response(house_x,data,'x')
modelx.NewmarkB()
modelx.FftMethod()
modelx.plot()

# modely = Response(house_y,data,'y')
# modely.NewmarkB()
# modely.FftMethod()
# modely.plot()

elapsed_time = time() - tstart
print(f'\nTotal time: {elapsed_time:.2f}s')