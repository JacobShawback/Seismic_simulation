from myPack.modal_analysis_method import ModalAnalysis
from myPack.data_processing import Data
from myPack.make_matrix import House_NL
import numpy as np
from time import time
from myPack.output_format import Format
from myPack.response_method import NL_4dof,NL_2dof
from myPack.constitution import Linear,Slip,Slip_Bilinear, Slip_Bilinear2
Format.params()

tstart = time()

# INPUT DATA --------------------------------------
data = Data("data/JR_Takatori_NS.acc",fname='ground/takatori',div=2**3)
# data.Output()
# data.ResponseSpectrum()


# STRUCTURE PARAMETER --------------------------------------
Area_f = 71.626  # [m2]
m1 = 9760.54  # 2F weight [kg]
m2 = 6439.62  # 1F weight [kg]
mf = 29198.65  # base weight [kg]
Ifx = 570.870/Area_f*mf  # moment of inertia [kg*m2]
Ify = 343.980/Area_f*mf
print('Ifx,Ify:',Ifx,Ify)
kc = 4420.76e3,3870.72e3  # [N/m]
kwx = 1659.21e3,2568.38e3
kwy = 1799.07e3,2482.34e3
l1,l2 = 5.8,3.0  # [m]
kh = 1.71e9  # [N/m]
kth = 3.03e10  # [N*m/rad]
cf = 2.976e6  # [N/m]
cth = 2.509e7

# args = {'m1':m1,'m2':m2,'mf':mf,'kh':kh,'kth':kth,'l1':l1,'l2':l2,'cf':cf,'cth':cth,'alpha_c':0.05,'alpha_w':0.24,'slip_rate_c':0.85,'slip_rate_w':0.85}
args = {'m1':m1,'m2':m2,'mf':mf,'kc':kc,'kh':kh,'kth':kth,'l1':l1,'l2':l2,'cf':cf,'cth':cth,'alpha_c':0.05,'alpha_w':0.24,'slip_rate_c':0.85,'slip_rate_w':0.85}
house_x = House_NL(**args,If=Ifx,kw=kwx)
# house_y = House_NL(**args,If=Ify,kw=kwy)


# # FREQ ANALYSIS -------------------------------------
cmodel = Slip_Bilinear2
# cmodel = Linear

modelx = NL_4dof(house_x,data,'x')
modelx.different(cmodel=cmodel)
modelx.plot(cmodel=cmodel)

# modely = NL_4dof(house_y,data,'y')
# modely.different(cmodel=cmodel)
# modely.plot(cmodel=cmodel,gif=True)

# second = modelx.get_u_angle(cmodel)
# second['tag1'],second['tag2'] = 'NS, ','EW, '
# modely.plot(cmodel=cmodel,second=second)

elapsed_time = time() - tstart
print(f'\nTotal time: {elapsed_time:.2f}s')