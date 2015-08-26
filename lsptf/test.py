'''
Created on 20 Aug 2015

@author: Oleg
'''
import numpy as np
from mapSignal import binData 

m = 10
n = 30
c = m*n*0.1

signal = np.random.randn(m,n)
linidx = np.random.choice(signal.size, c, replace=False)
np.put(signal,linidx,np.nan)

out = binData(signal,signal)
out = np.array(out)