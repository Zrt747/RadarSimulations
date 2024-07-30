import numpy as np
from joblib import Parallel, delayed
from .stft import stft

def Micro_Doppler(data,
                  wd, # window size
                  np, # number of pulses
                  use_parallel = False,
                  ):
    # window constants
    wdd2 = wd // 2
    wdd8 = wd // 8
    ns = np // wd  #number of segments

    if use_parallel:
        TF2 = parallel_stft_segments(data, wd, wdd8, ns)
        TF1 = parallel_shifted_segments(data, wd, wdd2, wdd8, ns)
    else:
        TF2 = stft_segments(data, wd, wdd8, ns)
        TF1 = shifted_segments(data, wd, wdd2, wdd8, ns)        
    return remove_edge_effects(TF2, TF1, wdd8, ns)


def process_segment(x, wd, wdd8, k):
    sig = x[k * wd : k * wd + wd]
    TMP = stft(sig, 16)
    return TMP[:, ::8]

def process_shifted_segment(x, wd, wdd2, wdd8, k):
    sig = x[k * wd + wdd2 : k * wd + wd + wdd2]
    TMP = stft(sig, 16)
    return TMP[:, ::8]

def parallel_stft_segments(x, wd, wdd8, ns):
    print('Calculating segments of Time-Frequency (TF) distribution ...')
    TF2 = np.zeros((wd, ns * wdd8), dtype=complex)
    results = Parallel(n_jobs=-1)(delayed(process_segment)(x, wd, wdd8, k) for k in range(ns))
    for TMP, k in results:
        TF2[:, k * wdd8 : k * wdd8 + wdd8] = TMP
    return TF2

def stft_segments(x, wd, wdd8, ns):
    print('Calculating segments of Time-Frequency (TF) distribution ...')
    TF2 = np.zeros((wd, ns * wdd8), dtype=complex)
    for k in range(ns):
        TF2[:, k * wdd8:(k) * wdd8+wdd8] = process_segment(x, wd, wdd8, k)
    return TF2

def parallel_shifted_segments(x, wd, wdd2, wdd8, ns):
    print('Calculating shifted segments of Time-Frequency (TF) distribution ...')
    TF1 = np.zeros_like(parallel_stft_segments(x, wd, wdd8, ns))
    results = Parallel(n_jobs=-1)(delayed(process_shifted_segment)(x, wd, wdd2, wdd8, k) for k in range(ns - 1))
    for TMP, k in results:
        TF1[:, k * wdd8 : k * wdd8 + wdd8] = TMP
    return TF1

def shifted_segments(x, wd, wdd2, wdd8, ns):
    print('Calculating shifted segments of Time-Frequency (TF) distribution ...')
    TF1 = np.zeros((wd, ns * wdd8), dtype=complex)
    for k in range(ns-1):
        TF1[:, k * wdd8 : k * wdd8 + wdd8] = process_shifted_segment(x, wd, wdd2, wdd8, k)
    return TF1

def remove_edge_effects(TF, TF1, wdd8, ns):
    print('Removing the edge effect ...')
    for k in range(ns - 1):
        TF[:, (k + 1) * wdd8 - 8 : (k + 1) * wdd8 + 8] = TF1[:, (k * wdd8) + (wdd8 // 2) - 8 : (k * wdd8) + (wdd8 // 2) + 8]
    return TF