import numpy as np
def exp_kernel(D, alpha):
    return  (alpha**2)/(2*np.pi) * np.exp(- alpha * D)

def exp_kernel_fft(ncols, nrows, alpha, pixel_size):
    di = np.arange(-(nrows - 1), nrows)
    dj = np.arange(-(ncols - 1), ncols)
    DI, DJ = np.meshgrid(di, dj, indexing='ij')
    return exp_kernel(pixel_size * np.sqrt(DI**2 + DJ**2), alpha) 

from scipy.signal import fftconvolve
def kernel_convolution(M, K, ncols, nrows):
    # ncols and nrows are the dimensions of M
    full = fftconvolve(M, K, mode='full')
    lam = full[nrows - 1:2*nrows - 1, ncols - 1:2*ncols - 1]
    return np.exp(-lam)