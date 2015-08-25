'''
Created on 20 Aug 2015

@author: Oleg
'''
import numpy as np
from mapSignal import _mapCore 

m = 5000
n = 9000
c = m*n*0.1

signal = np.random.randn(m,n)
linidx = np.random.choice(signal.size, c, replace=False)
np.put(signal,linidx,np.nan)

_mapCore(signal,PortfolioNumber=10)

