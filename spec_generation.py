from astropy.io import fits
import numpy as np
import glob
from glob import glob
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
import scipy.constants as C

def func(r, a, x, b):
    r = r * 1e-10
    return a * 8 * np.pi * C.h * C.c / r ** 5 / (np.exp(C.h * C.c / r / C.k / x) - 1) + b

def process(path, low_T, high_T):
    with fits.open(path) as hdu_list:
        spec = hdu_list[0].data

    flux = spec[0]
    wave = spec[2]

    flux = flux[: 3700]
    wave = wave[: 3700]
    flux = np.array(flux).astype(float)
    wave = np.array(wave).astype(float)
    fmed = median_filter(flux, size=299, mode='reflect')
    popt, _ = curve_fit(func, wave, fmed, bounds=([1e-8, low_T, - np.abs(np.min(fmed))], [1e-1, high_T, np.abs(np.min(fmed))+0.00000001]))
    curv = func(wave, *popt)
    fmax = np.max(curv)
    ferr = flux / fmax
    fstd = np.std(ferr)
    return wave, fmax, ferr, fstd, popt

def G_spectra(paths, numbers, flux, scls, star_type, high_T, low_T):
    wave_list = []
    fmax_list = []
    ferr_list = []
    fstd_list = []
    popt_list = []
    for path in glob(paths + '*.fits'):
        wave, fmax, ferr, fstd, popt = process(path, high_T, low_T)
        wave_list.append(wave)
        fmax_list.append(fmax)
        ferr_list.append(ferr)
        fstd_list.append(fstd)
        popt_list.append(popt)

    wave_list = np.array(wave_list)
    fmax_list = np.array(fmax_list)
    ferr_list = np.array(ferr_list)
    fstd_list = np.array(fstd_list)
    popt_list = np.array(popt_list)

    idx = np.argsort(fstd_list)[: np.int(0.5 * len(fstd_list))]

    wave_list = wave_list[idx]
    fmax_list = fmax_list[idx]
    ferr_list = ferr_list[idx]
    popt_list = popt_list[idx]

    a_list = popt_list[:, 0]
    s_list = popt_list[:, 2]

    for name in range(numbers):
        choice = np.random.randint(len(a_list), size=(5))
        wave = wave_list[choice[0]]
        fmax = fmax_list[choice[1]]
        ferr = ferr_list[choice[2]]
        a = a_list[choice[3]]
        b = np.random.random() * (low_T - high_T) + high_T
        s = s_list[choice[4]]
        curve = func(wave, a, b, s)
        f = curve * ferr
        f = (f - np.min(f)) / (np.max(f) - np.min(f))
        f = np.array([np.array([f])])
        flux.append(f)
        scls.append(star_type)
    return flux, scls

