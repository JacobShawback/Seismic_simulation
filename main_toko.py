from myPack.modal_analysis_method import ModalAnalysis
from myPack.data_processing import Data
from myPack.make_matrix import House_NL
import numpy as np
from time import time
from myPack.output_format import Format
from myPack.response_method import NL_4dof,NL_2dof
from myPack.constitution import Linear,Slip,Slip_Bilinear
Format.params()

tstart = time()

# INPUT DATA --------------------------------------
data = Data("data/JR_Takatori_NS.acc",fname='ground/takatori',div=2**4)
# data.Output()
# data.ResponseSpectrum()


# STRUCTURE PARAMETER --------------------------------------
Area_f = 71.626  # [m2]
m1 = 7553  # 2F weight [kg]
m2 = 10260
mf = 71053  # base weight [kg]
Ifx = 176000
kcx = 5177e3,5475e3
kwx = 1770e3,2731e3
l1,l2 = 6,3  # [m]
kh = 5.09e9  # [N/m]
kth = 1.67e10  # [N*m/rad]

# args = {'m1':m1,'m2':m2,'mf':mf,'kh':kh,'kth':kth,'l1':l1,'l2':l2,'cf':cf,'cth':cth,'alpha_c':0.05,'alpha_w':0.24,'slip_rate_c':0.85,'slip_rate_w':0.85}
args = {'m1':m1,'m2':m2,'mf':mf,'kh':kh,'kth':kth,'l1':l1,'l2':l2,'alpha_c':0.05,'alpha_w':0.24,'slip_rate_c':0.85,'slip_rate_w':0.85}
house_x = House_NL(**args,If=Ifx,kc=kcx,kw=kwx)
# house_y = House_NL(**args,If=Ify,kc=kcy,kw=kwy)


# # FREQ ANALYSIS -------------------------------------
cmodel = Slip
# modelx = NL_4dof(house_x,data,'x')
modelx = NL_4dof(house_x,data,'toko')
# modelx = NL_2dof(house_x,data,'2dof')
modelx.different(cmodel=cmodel)
modelx.plot(cmodel=cmodel)

elapsed_time = time() - tstart
print(f'\nTotal time: {elapsed_time:.2f}s')