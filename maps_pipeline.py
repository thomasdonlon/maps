import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from astropy.io import fits
from scipy.signal import argrelextrema
from scipy.optimize import differential_evolution, curve_fit
from scipy.fft import rfft, rfftfreq, irfft
from scipy.stats import linregress
from numpy import polyfit
from astropy.modeling import models, fitting
import ctypes
import multiprocessing as mp
from multiprocessing import Pool
import pickle
import wget

import warnings
warnings.filterwarnings('ignore')

pickle_dir = '/home/donlot/pickle/maps/'

# #all the fits files for the same star
# fits_filenames = [
#                   '/home/donlot/data/kepler_data/nr_lyrae/q1.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q2.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q3.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q4.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q5.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q6.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q7.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q8.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q9.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q10.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q11.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q12.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q13.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q14.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q15.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q16.fits',
#                   '/home/donlot/data/kepler_data/nr_lyrae/q17.fits',
#                  ]

# #load in all the data and build the dictionaries
# data = {}
# i = 0
# for f in fits_filenames:
    
#     hdul = fits.open(f)
#     splits = [] #context, this may not be required
    
#     #split the data up more
    
#     #define the splitting for each data period
#     len_hdul = len(hdul[1].data['time'])
#     if (i != 0) and (f != fits_filenames[-1]): #skip the first and the last, they're already smaller
#         n_splits = 8
#         splits = [np.arange(j*len_hdul//n_splits, (j+1)*len_hdul//n_splits) for j in range(n_splits)]
#     else:
#         splits = [np.arange(0, len_hdul)]
    
#     for split in splits: #there will only be 1 split if you don't need to split up that data
        
#         # Read in the "BJDREF" which is the time offset of the time array.
#         bjdrefi = hdul[1].header['BJDREFI']
#         bjdreff = hdul[1].header['BJDREFF']

#         # Read in the columns of data.
#         times = hdul[1].data['time'][split]
#         sap_fluxes = hdul[1].data['SAP_FLUX'][split]
#         pdcsap_fluxes = hdul[1].data['PDCSAP_FLUX'][split]

#         bjds = times + bjdrefi + bjdreff 

#         sample_spacing = (bjds[-1] - bjds[0])/len(bjds)
#         sample_rate = 1/sample_spacing

#         data[i] = {
#                    'bjds': bjds,
#                    'sap_fluxes': sap_fluxes,
#                    'pdcsap_fluxes': pdcsap_fluxes,
#                    'sample_spacing': sample_spacing,
#                    'sample_rate': sample_rate
#                   }
        
#         i += 1
    
# print('data read in successfully')

#load in all the data and build the dictionaries
data = {}
#n_splits = 6 #number of times to split each segment of data
len_segment = 165 #minimum number of stars to put in one segment
n_splits_first = 0

#kld settings
n_time_bins = 5 #how many time bins to split up time segments (~5 seems to work well)
n_kld_bins = 50 #how many bins are used to compute KLD (sensitive to choice, not really a true numerical integral)
kld_scale = 1 #how much to weight the kld value (should probably be a fractional amount rather than scalar multiple)

#download the data 
#filename = '/home/donlot/data/kepler_data/kepler_lc_wget_V2367_Cyg.txt' #V2367_Cyg
filename = '/home/donlot/data/kepler_data/kepler_lc_wget_7106205.txt' #KIC 7106205

id_tracker = 0
#should do splits as a uniform amount of time, not n splits per data set
with open(filename) as f:
    for i, line in enumerate(f):
        fits_file = wget.download(line.lstrip('wget --no-verbose '))
    
        hdul = fits.open(fits_file)

        splits = [] #context, this may not be required
        
        #define the splitting for each data period
        len_hdul = len(hdul[1].data['time'])
        n_splits = int(np.floor(len_hdul/len_segment))
        if i == 0:
            n_splits_first = n_splits
        splits = [np.arange(j*len_hdul//n_splits, (j+1)*len_hdul//n_splits) for j in range(n_splits)]

        for split in splits: #there will only be 1 split if you don't need to split up that data
            
            # Read in the "BJDREF" which is the time offset of the time array.
            bjdrefi = hdul[1].header['BJDREFI']
            bjdreff = hdul[1].header['BJDREFF']

            # Read in the columns of data.
            times = hdul[1].data['time'][split]
            sap_fluxes = hdul[1].data['SAP_FLUX'][split]
            pdcsap_fluxes = hdul[1].data['PDCSAP_FLUX'][split]

            bjds = times + bjdrefi + bjdreff 

            tmp_sample_spacing = (bjds[-1] - bjds[0])/len(bjds)
            tmp_sample_rate = 1/tmp_sample_spacing
            
            if not np.isnan(tmp_sample_spacing):
                sample_spacing = tmp_sample_spacing
            if not np.isnan(tmp_sample_rate):
                sample_rate = tmp_sample_rate

            data[id_tracker] = {
                       'bjds': bjds,
                       'sap_fluxes': sap_fluxes,
                       'pdcsap_fluxes': pdcsap_fluxes,
                       'sample_spacing': sample_spacing,
                       'sample_rate': sample_rate
                      }
            
            id_tracker += 1
    
print('data read in successfully')

#do a fft and find the fundamental frequency

#use all data from the first segment
pdcsap_fluxes = np.array([])
for i in range(n_splits_first+1): #context hell
    pdcsap_fluxes = np.append(pdcsap_fluxes, data[i]['pdcsap_fluxes'])

#--------------------
#have to fix the fluxes first to actually get an fft

#fix edges if they are nan
if np.isnan(pdcsap_fluxes[-1]):
    pdcsap_fluxes[-1] = np.nanmean(pdcsap_fluxes)
if np.isnan(pdcsap_fluxes[0]):
    pdcsap_fluxes[0] = np.nanmean(pdcsap_fluxes)


interp_pdcsap_fluxes = [(pdcsap_fluxes[i] if not(np.isnan(pdcsap_fluxes[i])) else (pdcsap_fluxes[i-1] + pdcsap_fluxes[i+1])/2 ) for i in range(len(pdcsap_fluxes))]

for j in range(100):
    interp_pdcsap_fluxes = [(interp_pdcsap_fluxes[i] if not(np.isnan(interp_pdcsap_fluxes[i])) else (interp_pdcsap_fluxes[i-j] + interp_pdcsap_fluxes[i+j])/2 ) for i in range(len(pdcsap_fluxes))]
    
i = 0
while i < len(interp_pdcsap_fluxes) and not(np.isnan(interp_pdcsap_fluxes[i])):
    i += 1
interp_pdcsap_fluxes = interp_pdcsap_fluxes[:i]
#-------------------

#compute the fft
tmp = interp_pdcsap_fluxes - np.mean(interp_pdcsap_fluxes)
power_series = rfft(tmp / np.max(tmp))
mag_power_series = np.abs(power_series)
power_series_bins = rfftfreq(len(interp_pdcsap_fluxes), sample_spacing)

delta_f = sample_rate / len(pdcsap_fluxes)
print('scale of fft:', delta_f, ' inverse BJDs')

n_skip = 10 #how many bins to skip on the left of the fft
#(for messy light curves, the smallest bins of the fft are artificially inflated)

#calculate the period
ffreq = power_series_bins[n_skip:][np.argmax(mag_power_series[n_skip:])]
period = 1/ffreq #BJDs, technically a period guess
print('fund. freq.:', ffreq, 'days^-1')
print('period:', period, 'days')
period = 0.0092 #hard coded for this particular star

scan_width = 0.000001
scan_n = 1e3
times = np.arange(period-scan_n*scan_width, period+scan_n*scan_width, scan_width)

def kld(arr):
    h = np.histogram(arr, bins=n_kld_bins)[0]
    kld = h * np.log(h/np.mean(h))
    return np.nansum(kld)

#the optimizer
def likelihood(period, data, verbose_out=False):
   
    #fold the data based on the period
    folded_bjds = (data['bjds'] % period)/period

    '''
    #scale the fluxes so they have min = 0, max = 1 in each period
    #(attempt to remove amplitude modulation)
    n_periods = 1 #should probably stay 1, more = more periods in each scaling calculation
                  #and this can cause issues if the amp. mod. period is on the order of 1 period
    n_wraps = (data['bjds']//(n_periods * period))
    n_wraps = np.floor(n_wraps - np.nanmin(n_wraps)).astype(int)

    #this handles nans
    if np.min(n_wraps) < 0: #nans as ints will be like negative 4 billion
        remove_indx = np.where(n_wraps == np.min(n_wraps))[0]
        n_wraps = np.delete(n_wraps, remove_indx)
        flux_data = np.delete(data['pdcsap_fluxes'], remove_indx)
        folded_bjds = np.delete(folded_bjds, remove_indx)
    else:
        remove_indx = np.array([])
        flux_data = data['pdcsap_fluxes']

    fixed_fluxes = np.array([])

    if len(n_wraps) == 0:
        return 1e10
    for i in range(np.max(n_wraps)+1):
        wrap_indx = (n_wraps == i)
        new_fluxes = flux_data[wrap_indx]
        if len(new_fluxes) > 0:
            new_fluxes = new_fluxes - np.nanmin(new_fluxes)
            new_fluxes = new_fluxes / np.nanmax(new_fluxes)
        
        fixed_fluxes = np.append(fixed_fluxes, new_fluxes)
    '''
    fixed_fluxes = data['pdcsap_fluxes']
    
    #bin the data
    bin_width = 0.01
    binned_pdcsap_fluxes = []
    binned_bjds = []
    
    i = 0
    while i < 1:
        indx = (folded_bjds >= i)*(folded_bjds < i+bin_width)
        binned_pdcsap_fluxes.append(fixed_fluxes[indx])
        binned_bjds.append(folded_bjds[indx])
        i += bin_width
    
    #compute the dispersion of each bin w.r.t. a linear fit:    
    #then adjust the data according to that linear fit
    slope_adj_fluxes = []
    lfs = []
    for i in range(len(binned_bjds)):
        indx = (np.logical_not(np.isnan(binned_bjds[i])))*(np.logical_not(np.isnan(binned_pdcsap_fluxes[i])))
        if len(binned_pdcsap_fluxes[i][indx]) == 0:
            slope_adj_fluxes.append([np.nan, np.nan, np.nan])
        else:
            lf = linregress(binned_bjds[i][indx], binned_pdcsap_fluxes[i][indx])
            lfs.append(lf)
            slope_adj_fluxes.append([binned_pdcsap_fluxes[i][indx][j] - (lf[0]*binned_bjds[i][indx][j] + lf[1]) for j in range(len(binned_pdcsap_fluxes[i][indx]))])
    
    disps = np.array([np.nanstd(f) for f in slope_adj_fluxes])
    for i in range(len(disps)):
        if len(binned_pdcsap_fluxes[i]) == 0:
            disps[i] = np.nan

    #compute the KLD of the binned folded times 
    #bin the data
    bin_width = (np.max(data['bjds']) - np.min(data['bjds']))/n_time_bins
    centered_times = data['bjds'] - np.min(data['bjds'])
    mean_kld = 0
    for i in range(n_time_bins):
        indx = (centered_times >= i*bin_width)*(centered_times < (i+1)*bin_width)
        mean_kld += kld(folded_bjds[indx])
    mean_kld /= n_time_bins
    
    #likelihood score is just the sum of all dispersions + scaled KLD of binned folded bjds
    penalty = 0 #penalize not having anything in a bin
    like = np.nansum(disps**2)/len(disps[np.logical_not(np.isnan(disps))]) + len(disps[np.isnan(disps)])*penalty + kld_scale*mean_kld
    
    if verbose_out:
        return like, disps, lfs
    else:
        return like

#for threading
def worker(key):
    print('starting thread', key)
    all_likes = []

    best_period = period
    best_like = likelihood(period, data[key])

    for p in times:
        like = likelihood(p, data[key])
        all_likes.append(like)
        if like < best_like:
            best_period = p
            best_like = like

    print('ending thread', key)
    return key, best_period, all_likes

#brute force bc the likelihood surface isn't actually differentiable
#now with threading
def scan_time_slices(data):
    
    times = np.arange(period-scan_n*scan_width, period+scan_n*scan_width, scan_width)
    
    #create the helper db for the threads
    #ldb = ListDatabase(len(data.keys()))
    best_periods = [0]*len(data.keys())
    all_like_list = [0]*len(data.keys())

    with Pool(125) as pool:
        result = pool.map(worker, data.keys())

    for r in result:
    	best_periods[r[0]] = r[1]
    	all_like_list[r[0]] = r[2]
        
    return best_periods, all_like_list

#optimize the period of each time slice
best_periods, all_like_list = scan_time_slices(data)

#dump the low-res scan data (can analyze this elsewhere)
with open(pickle_dir + 'best_periods_lowres_ds.pickle', 'wb') as p:
	pickle.dump(best_periods, p)

with open(pickle_dir + 'all_like_list_lowres_ds.pickle', 'wb') as p:
	pickle.dump(all_like_list, p)

#find the "mean minimum" of all the likelihood surfaces
min_indx_list = []
for key in data.keys():
    min_indx = np.argmin(all_like_list[key])
    min_indx_list.append(min_indx)
mean_min_indx = int(np.mean(min_indx_list))

#re-run the parameter sweep, but on high-resolution now
scan_width = 0.0000005
scan_n = 1e3
period = times[mean_min_indx]
times = np.arange(period-scan_n*scan_width, period+scan_n*scan_width, scan_width)

#optimize the period of each time slice
best_periods, all_like_list = scan_time_slices(data)

#dump the high-res scan data (can analyze this elsewhere)
with open(pickle_dir + 'best_periods_highres_ds.pickle', 'wb') as p:
	pickle.dump(best_periods, p)

with open(pickle_dir + 'all_like_list_highres_ds.pickle', 'wb') as p:
	pickle.dump(all_like_list, p)