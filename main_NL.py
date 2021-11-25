# from myPack.response_method import Response
from myPack.modal_analysis_method import ModalAnalysis
from myPack.data_processing import Data
from myPack.make_matrix import House_NL
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
from myPack.response_method import NL_4dof,NL_2dof
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
# k1x = 6079.97e3  # [N/m]
# k2x = 6439.10e3
kcx = 3000e3,3000e3
kwx = 3080e3,3440e3
# k1y = 6219.83e3
# k2y = 6353.06e3
kcy = 3000e3,3000e3
kwy = 3220e3,3350e3
l1,l2 = 5.8,3.0  # [m]
kh = 2.83e9  # [N/m]
kth = 5.75e10  # [N*m/rad]
cf = 4.38e6  # [N/m]
cth = 3.69e7

args = {'m1':m1,'m2':m2,'mf':mf,'kh':kh,'kth':kth,'l1':l1,'l2':l2,'cf':cf,'cth':cth,'clough_alpha_c':0.05,'clough_alpha_w':0.05}
# args = {'m1':m1,'m2':m2,'mf':mf,'kh':kh,'kth':kth,'l1':l1,'l2':l2,'cf':cf,'cth':cth,'clough_alpha_c':1,'clough_alpha_w':1}
house_x = House_NL(**args,If=Ifx,kc=kcx,kw=kwx)
# house_y = House_NL(m1,m2,mf,Ify,kcy,kwy,kh,kth,l1,l2,cf=cf,cth=cth)


# INPUT DATA --------------------------------------
data = Data("data/JR_Takatori_NS.acc",fname='ground/takatori',div=2**1)
# data.Output()
# data.ResponseSpectrum()


# # MODAL ANALYSIS --------------------------------------
# Mode = ModalAnalysis(house_x.M, house_x.K, house_x.F)
# Mode.NaturalValue('4DOF')
# # Mode.Participation()
# # print(np.sum(Mode.beta))


# # FREQ ANALYSIS -------------------------------------
modelx = NL_2dof(house_x,data,'nl-x-nb')
modelx.newmark()
# modelx = NL_2dof(house_x,data,'nl-x-df')
# modelx = NL_4dof(house_x,data,'nl-x-df-4')
# modelx.different()
modelx.plot()

# modely = Response(house_y,data,'y')
# modely.NewmarkB()
# modely.FftMethod()
# modely.plot()

elapsed_time = time() - tstart
print(f'\nTotal time: {elapsed_time:.2f}s')