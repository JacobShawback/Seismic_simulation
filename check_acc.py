from myPack.modal_analysis_method import ModalAnalysis
from myPack.data_processing import Data
from myPack.make_matrix import House_NL
import numpy as np
from time import time
from myPack.output_format import Format
Format.params()

tstart = time()

# INPUT DATA --------------------------------------
data = Data("mySGF/data/acc0.txt",fname='ground/ikoma',div=1)
data.Output()
data.ResponseSpectrum()

data = Data("mySGF/data/egf_acc.txt",fname='ground/egf',div=1)
data.Output()
data.ResponseSpectrum()