from myPack.modal_analysis_method import ModalAnalysis
from myPack.data_processing import Data
from myPack.make_matrix import House_NL
import numpy as np
from time import time
from myPack.output_format import Format
Format.params()

tstart = time()

# INPUT DATA --------------------------------------
# before = Data("mySGF/data/acc0.txt",fname='ground/ikoma',div=1)
# before.Output()
# data.ResponseSpectrum()

takatori = Data("data/JR_Takatori_NS.acc",fname='ground/takatori',div=1)
# takatori.Output()
Sd_takatori = takatori.ResponseSpectrum()

data = Data("mySGF/data/egf_acc.txt",fname='ground/egf',div=1)
# data.Output(before.acc0)
data.ResponseSpectrum(Sd_takatori)