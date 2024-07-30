import numpy as np
import matplotlib.pyplot as plt

def is_jupyter():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except:
        return False
    return True

def calculate_power_range(return_power_db,help = False):

    return_power_db = return_power_db[~np.isnan(return_power_db)]

    if help:
        fig, ax = plt.subplots()
        ax.hist(return_power_db)
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Return Power (dB)')
        ax.set_title('Average Return Power vs. Range')
        ax.grid(True)
        plt.show()

    hist, bin_edges = np.histogram(return_power_db, bins=50)
    threshold_index = otsu_threshold(hist)
    otsu_threshold_value = bin_edges[threshold_index+1]

    # remove threshold
    new_power = return_power_db[return_power_db>otsu_threshold_value]
    return [np.min(new_power), np.max(new_power)]


def otsu_threshold(hist):
    total = np.sum(hist)
    current_max, threshold = 0, 0
    sumB, wB = 0, 0
    sum1 = np.dot(np.arange(len(hist)), hist)

    for i in range(len(hist)):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between > current_max:
            current_max = between
            threshold = i

    return threshold