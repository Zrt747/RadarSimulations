import numpy as np
from scipy.fftpack import fft

def stft(f, wd):
    """
    Short-time Fourier transform with Gaussian window
    
    Parameters:
    f - signal (real or complex)
    wd - standard deviation (sigma) of the Gaussian function
    
    Returns:
    TF - time-frequency distribution
    """
    cntr = len(f) // 2
    sigma = wd
    fsz = len(f)

    # Gaussian window calculation
    # z = np.exp(-((np.arange(1, fsz + 1) - fsz / 2) ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
    z = np.exp(-((np.arange(fsz) - cntr) ** 2)           / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


    gwin = np.roll(z, shift=-cntr + fsz // 2)
    # gwin = np.zeros(fsz)
    # for m in range(fsz):
    #     mm = m - cntr + fsz // 2
    #     if 1 <= mm <= fsz:
    #         gwin[m] = z[mm - 1]
    #     elif mm > fsz:
    #         gwin[m] = z[mm % fsz - 1]
    #     elif mm < 1:
    #         gwin[m] = z[mm + fsz - 1]

    # Zero padding
    winsz = len(gwin)
    x = np.zeros(fsz + winsz, dtype=complex)  # zero padding
    x[winsz // 2 : winsz // 2 + fsz ] = f
    
    TF = np.zeros((winsz, fsz), dtype=complex)
    for j in range(fsz):
        X = gwin * x[j : j + winsz]
        TF[:, j] = fft(X)

    return TF
