'''
Created on 20 Aug 2015

@author: Oleg
'''
import numpy as np
from mapSignal import binData 

m = 1000
n = 200
c = m*n*0.1

signal = np.random.randn(m, n)
linidx = np.random.choice(signal.size, c, replace=False)
np.put(signal, linidx, np.nan)
np.savetxt("signal.csv", signal, delimiter=",")

out = binData(signal,signal, IndependentSort=False)

np.savetxt("out1.csv", out[0], delimiter=",")
np.savetxt("out2.csv", out[1], delimiter=",")