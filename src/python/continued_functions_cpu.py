# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1aBQmd5CpJc16C0bCwiyleOF2a4ad8NO0
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def continued_fraction(x, terms):
    val = x
    for i in range(terms):
        if i == terms-1:
            val = 1/(1-val)
        else:
            val = x/(1-val)
        # print(val)
    return val

def continued_exp(x, terms):
    val = 1
    for i in range(terms):
        val = np.exp(val*x)
        # print(val)
    return val

N = 199
conv = 50
df = 0.01
s = np.zeros((N,N))
pt = np.zeros((conv,),dtype=np.complex_)
for x in range(N):
    for y in range(N):
        dx = x * df - N * df/2
        dy = y * df - N * df/2
        for i in range(conv):
            pt[i] = continued_exp(dx+dy*1j, i)
        s[x,y] = np.unique(np.array(pt)).size

plt.figure(figsize=(8,8))
plt.set_cmap('hot')
plt.imshow(s)
