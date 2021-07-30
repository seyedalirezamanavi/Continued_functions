# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1aBQmd5CpJc16C0bCwiyleOF2a4ad8NO0
"""

import numpy as np
import matplotlib.pyplot as plt

import cmath
import math
from numba import cuda

# the continued exponential function cuda kernel
@cuda.jit
def number_of_LP(N, conv, pt, df, fst_x, fst_y):
    bwx = cuda.blockDim.x
    bId_x = bwx * cuda.blockIdx.x
    bwy = cuda.blockDim.y
    bId_y = bwy * cuda.blockIdx.y
    tx = cuda.threadIdx.x + bId_x
    ty = cuda.threadIdx.y + bId_y


    dx = tx * df + fst_x
    dy = ty * df + fst_y
    z = dx+dy*1j
    for terms in range(conv):

        val = 1
        for k in range(10000,10000+terms):
            val = cmath.exp(val*z)
        pt[tx, ty, terms] = val



# the continued exponential function call section


n = 1024
N = 64
conv = 300
df = 0.005
pt = np.zeros((N, N, conv), dtype=np.complex64)
threadsperblock = (32, 32)
blockspergrid_x = math.ceil(N / threadsperblock[0])
blockspergrid_y = math.ceil(N / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
cp = np.zeros((n,n))

for fx in range(n//N):
    for fy in range(n//N):
        ofst_x = fx * N * df - n * df/2
        ofst_y = fy * N * df - n * df/2
        number_of_LP[blockspergrid, threadsperblock](N, conv, pt, df, ofst_x, ofst_y)
        for i in range(N):
            for j in range(N):
                cp[i+N*fx,j+N*fy] = np.unique(np.round(pt.reshape(N,N, conv)[i,j,:],decimals=4)).size

plt.figure(figsize = (10,10))
plt.set_cmap('hot')

plt.imshow(cp)
plt.show()


# the continued fraction function cuda kernel

@cuda.jit
def number_of_LP(N, conv, pt, df, fst_x, fst_y):
    bwx = cuda.blockDim.x
    bId_x = bwx * cuda.blockIdx.x
    bwy = cuda.blockDim.y
    bId_y = bwy * cuda.blockIdx.y
    tx = cuda.threadIdx.x + bId_x
    ty = cuda.threadIdx.y + bId_y


    dx = tx * df + fst_x
    dy = ty * df + fst_y
    z = dx+dy*1j
    for terms in range(conv):
        # pt[index, i] = continued_exp(dx+dy*1j, i)

        val = z
        for i in range(terms):
            if i == terms-1:
                val = 1/(1-val)
            else:
                val = z/(1-val)
        pt[tx, ty, terms] = val




# the continued fraction function call section

n = 512
N = 64
conv = 1000
df = 1000
pt = np.zeros((N, N, conv), dtype=np.complex64)
threadsperblock = (32, 32)
blockspergrid_x = math.ceil(N / threadsperblock[0])
blockspergrid_y = math.ceil(N / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)
cp = np.zeros((n,n))

for fx in range(n//N):
    for fy in range(n//N):
        ofst_x = fx * N * df - n * df/2
        ofst_y = fy * N * df - n * df/2
        number_of_LP[blockspergrid, threadsperblock](N, conv, pt, df, ofst_x, ofst_y)
        for i in range(N):
            for j in range(N):
                cp[i+N*fx,j+N*fy] = np.unique(np.round(pt.reshape(N,N, conv)[i,j,:],decimals=4)).size


plt.figure(figsize = (10,10))
plt.set_cmap('hot')

plt.imshow(cp)
plt.show()
