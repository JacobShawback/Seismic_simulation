from myPack.modal_analysis_method import ModalAnalysis
from myPack.data_processing import Data
from myPack.make_matrix import House_NL
import numpy as np
from time import time
from myPack.output_format import Format
from myPack.response_method import NL_4dof,NL_2dof
Format.params()

tstart = time()

# INPUT DATA --------------------------------------
data = Data("data/JR_Takatori_NS.acc",fname='ground/takatori',div=2**0)
# data.Output()
# data.ResponseSpectrum()


# STRUCTURE PARAMETER --------------------------------------
Area_f = 71.626  # [m2]
m1 = 9760.54  # 2F weight [kg]
m2 = 6439.62  # 1F weight [kg]
mf = 29198.65  # base weight [kg]
Ifx = 570.870/Area_f*mf  # moment of inertia [kg*m2]
Ify = 343.980/Area_f*mf
kcx = 4420.6e3,3870.72e3  # [N/m]
kwx = 1659.21e3,2568.38e3
kcy = 4420.76e3,3870.72e3
kwy = 1659.21e3,2482.34
l1,l2 = 5.8,3.0  # [m]
kh = 2.83e9  # [N/m]
kth = 5.75e10  # [N*m/rad]
cf = 4.38e6  # [N/m]
cth = 3.69e7

args = {'m1':m1,'m2':m2,'mf':mf,'kh':kh,'kth':kth,'l1':l1,'l2':l2,'cf':cf,'cth':cth,'alpha_c':0.05,'alpha_w':0.05,'slip_rate_c':0.85,'slip_rate_w':0.85}
# args = {'m1':m1,'m2':m2,'mf':mf,'kh':kh,'kth':kth,'l1':l1,'l2':l2,'cf':cf,'cth':cth,'alpha_c':0.05,'alpha_w':0.05,'slip_rate_c':1,'slip_rate_w':1}
house_x = House_NL(**args,If=Ifx,kc=kcx,kw=kwx)
# house_y = House_NL(**args,If=Ify,kc=kcy,kw=kwy)


# # FREQ ANALYSIS -------------------------------------
linear = True
# modelx = NL_2dof(house_x,data,'x-nb-2-sb')
# modelx.newmark(linear=linear)
modelx = NL_2dof(house_x,data,'x-df-2-sb')  # slip-bilinear
# modelx = NL_2dof(house_x,data,'x-df-2-s')  # slip
# modelx = NL_4dof(house_x,data,'x-df-4')
modelx.different(linear=linear)
modelx.plot(linear=linear)

elapsed_time = time() - tstart
print(f'\nTotal time: {elapsed_time:.2f}s')