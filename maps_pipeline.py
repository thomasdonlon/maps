import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from numpy import polyfit
import multiprocessing as mp
from multiprocessing import Pool
import pickle

import warnings
warnings.filterwarnings('ignore') #good programming

#------------------------------------------------------------------
# SETTINGS (Change before running)
#------------------------------------------------------------------
#star files settings
base_filename = '/home/donlot/data/kepler_data/KIC11802860/'
n_files = 15
pickle_dir = '/home/donlot/pickle/maps/'
pickle_star_name = 'KIC11802860' #used for pickling the data

#star information
period = 0.687216

#algorithm settings
len_segment = 600 #number of measurements to put in one segment
scan_shift = 11 #number of measurements to shift between scans
n_poly = 4 #highest order for the polynomial fit in each bin
           #    4 is a good start. For weird light curves, a higher order may be necessary, but
           #    if the available degrees of freedom become small (large n_poly),
           #    the chi squared calculation may become inaccurate
scan_width = 0.0000001 #dt of a step in the scanning algorithm
scan_n = 2e3 #total number of likelihood function calculations, scan_n/2 in each direction from period
penalty = 0 #penalize not having anything in a bin, 
            #   this should probably be 0 but you can play around with it if you want
stitch_factor = 10 #if the difference between two points on the likelihood surface are 
                   #    >= stitch_factor*median difference between points on the likelihood surface to count,
                   #    then the algorithm stitches together the likelihood surface at that point. 
                   #    10 is a good place to start. 
#------------------------------------------------------------------

fits_filenames = [f"{base_filename}lc{n}.fits" for n in range(n_files)]

#load in all the data and build the dictionaries
data = {}

id_tracker = 0
#should do splits as a uniform amount of time, not n splits per data set
for i, fits_file in enumerate(fits_filenames):
    
    hdul = fits.open(fits_file)

    len_hdul = len(hdul[1].data['time'])

    start = 0
    while start + len_segment < len_hdul: #scan until you reach the end of the file

        # Read in the "BJDREF" which is the time offset of the time array.
        bjdrefi = hdul[1].header['BJDREFI']
        bjdreff = hdul[1].header['BJDREFF']

        # Read in the columns of data.
        times = hdul[1].data['time'][start:start+len_segment]
        sap_fluxes = hdul[1].data['SAP_FLUX'][start:start+len_segment]
        pdcsap_fluxes = hdul[1].data['PDCSAP_FLUX'][start:start+len_segment]
        pdcsap_err = hdul[1].data['PDCSAP_FLUX_ERR'][start:start+len_segment]

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
                   'pdcsap_err': pdcsap_err,
                   'sample_spacing': sample_spacing,
                   'sample_rate': sample_rate
                  }

        id_tracker += 1
        start += scan_shift
    
print('data read in successfully')
print(len(data.keys()), 'scans' )


times = np.arange(period-scan_n*scan_width, period+scan_n*scan_width, scan_width)

#the optimizer
def likelihood(period, data, verbose_out=False):

    #scale the fluxes (not necesarry but it gives the "correct" scaling on the likelihood)
    fixed_fluxes = data['pdcsap_fluxes']
    fixed_errs = data['pdcsap_err']
    
    fixed_fluxes = fixed_fluxes - min(fixed_fluxes) #doesn't change the relative errors
    
    fixed_errs = fixed_errs / max(fixed_fluxes)
    fixed_fluxes = fixed_fluxes / max(fixed_fluxes)

    #fold the data based on the period
    folded_bjds = (data['bjds']/period) - (data['bjds']/period).astype(int)

    #shift the folded times so that the max value is always in the first spot
    tmp = fixed_fluxes.copy()
    tmp[np.isnan(tmp)] = 0
    max_ind = np.argmax(tmp)
    folded_bjds = folded_bjds - folded_bjds[max_ind]
    for i, el in enumerate(folded_bjds):
        if el < 0:
            folded_bjds[i] += 1

    #bin the data
    bin_width = 0.025
    overlap = 1.0 #fraction of bin width to overlap for the fit
    fit_binned_pdcsap_fluxes = []
    fit_binned_bjds = []
    binned_pdcsap_fluxes = []
    binned_bjds = []
    binned_errs = []

    i = 0
    while i < 1:
        if i == 0: #have to wrap around for first and last bins
            indx = ((folded_bjds >= i-overlap*bin_width)*(folded_bjds < i+(1+overlap)*bin_width)) + (folded_bjds >= 1-overlap*bin_width)
            #wrap the data correctly
            tmp_folded_bjds = folded_bjds[indx]
            for j in range(len(tmp_folded_bjds)):
                if tmp_folded_bjds[j] > 0.5:
                    tmp_folded_bjds[j] -= 1
        elif i >= 1-bin_width:
            indx = ((folded_bjds >= i-overlap*bin_width)*(folded_bjds < i+(1+overlap)*bin_width)) + (folded_bjds <= overlap*bin_width)
            #wrap the data correctly
            tmp_folded_bjds = folded_bjds[indx]
            for j in range(len(tmp_folded_bjds)):
                if tmp_folded_bjds[j] < 0.5:
                    tmp_folded_bjds[j] += 1
        else:
            indx = (folded_bjds >= i-overlap*bin_width)*(folded_bjds < i+(1+overlap)*bin_width)
            tmp_folded_bjds = folded_bjds[indx]
        fit_binned_bjds.append(tmp_folded_bjds)
        fit_binned_pdcsap_fluxes.append(fixed_fluxes[indx])

        indx = (folded_bjds >= i)*(folded_bjds < i+bin_width)
        binned_pdcsap_fluxes.append(fixed_fluxes[indx])
        binned_bjds.append(folded_bjds[indx])
        binned_errs.append(fixed_errs[indx])

        i += bin_width

    #compute the dispersion of each bin w.r.t. a linear fit:    
    #then adjust the data according to that linear fit
    x2s = []
    for i in range(len(binned_bjds)):
        fit_indx = (np.logical_not(np.isnan(fit_binned_bjds[i])))*(np.logical_not(np.isnan(fit_binned_pdcsap_fluxes[i])))
        indx = (np.logical_not(np.isnan(binned_bjds[i])))*(np.logical_not(np.isnan(binned_pdcsap_fluxes[i])))
        if len(binned_pdcsap_fluxes[i][indx]) <= n_poly + 1: #can't compute a chi-squared for this
            x2s.append(np.nan)
        else:
            #poly fit
            popt = np.polyfit(fit_binned_bjds[i][fit_indx], fit_binned_pdcsap_fluxes[i][fit_indx], n_poly)
            y = np.polyval(popt, binned_bjds[i][indx])
            x2 = np.nansum((((binned_pdcsap_fluxes[i][indx] - y)/binned_errs[i][indx])**2))
            x2 = x2 / (len(binned_pdcsap_fluxes[i][indx]) - (n_poly + 1))
            x2s.append(x2)

    x2s = np.array(x2s)#[sum_ids]

    #likelihood score is just the sum of all the x2s
    like = np.nansum(x2s)/len(x2s[np.logical_not(np.isnan(x2s))]) + len(x2s[np.isnan(x2s)])*penalty

    if verbose_out:
        return np.log(like), x2s, lfs
    else:
        return np.log(like)

def stitch_data(arr, tol):
    out = arr.copy()
    for i in range(len(out)):
        if i == 0:
            pass
        else:
            diff = out[i] - out[i-1]
            if abs(diff) > tol:
                out[i:] = out[i:] - diff
    return out

#for threading
def worker(key):
    print('starting thread', key)

    all_likes = []

    for p in times:
        like = likelihood(p, data[key])
        all_likes.append(like)

    all_likes = np.array(all_likes)
    all_likes = stitch_data(all_likes, stitch_factor*np.median(np.abs(all_likes[1:] - all_likes[:-1])))

    best_like = min(all_likes)
    best_like_ind = np.argmin(all_likes)
    best_period = times[best_like_ind]

    print('ending thread', key)
    return key, best_period, all_likes

#brute force bc the likelihood surface isn't actually differentiable
#now with threading
def scan_time_slices(data):
    
    best_periods = [0]*len(data.keys())
    all_like_list = [0]*len(data.keys())

    with Pool(125) as pool:
        result = pool.map(worker, data.keys())

    for r in result:
    	best_periods[r[0]] = r[1]
    	all_like_list[r[0]] = list(r[2])
        
    return best_periods, all_like_list

#optimize the period of each time slice
best_periods, all_like_list = scan_time_slices(data)

#dump the scan data (can analyze this elsewhere)
with open(f"{pickle_dir}best_periods_{pickle_star_name}_ls{len_segment}_ss{scan_shift}.pickle", 'wb') as p:
	pickle.dump(best_periods, p)

with open(f"{pickle_dir}all_like_list_{pickle_star_name}_ls{len_segment}_ss{scan_shift}.pickle", 'wb') as p:
	pickle.dump(all_like_list, p)
