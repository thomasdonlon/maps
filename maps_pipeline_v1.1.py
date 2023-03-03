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
#star_id = '007988343'
#filename = f"/home/donlot/data/kepler_data/nonBL/table_kplr{star_id}.tailor-made.dat"
pickle_dir = '/home/donlot/pickle/maps/maps_v1.1/'

#star information
#period = 0.5811436

#algorithm settings
len_segment = 600 #number of measurements to put in one segment
scan_shift = 15 #number of measurements to shift between scans
n_poly = 7 #highest order for the polynomial fit in each bin
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
# CONSTANTS (Change before running)
#------------------------------------------------------------------

star_ids_list = [
                '003733346',
                '003866709',
                '005299596',
                '006070714',
                '006100702',
                '006763132',
                '006936115',
                '007030715',
                '007176080',
                '007742534',
                '007988343',
                '008344381',
                '009591503',
                '009658012',
                '009717032',
                '009947026',
                '010136240',
                '010136603',
                '011802860'
                ]

periods_list = [
               0.6820264,
               0.47070609,
               0.5236377,
               0.5340941,
               0.4881457,
               0.5877887,
               0.52739847,
               0.68361247,
               0.507074,
               0.4564851,
               0.5811436,
               0.5768288,
               0.5713866,
               0.533206,
               0.5569092,
               0.5485905,
               0.5657781,
               0.4337747,
               0.687216
              ]

#------------------------------------------------------------------

for star_id, period in zip(star_ids_list, periods_list):

    filename = f"/home/donlot/data/kepler_data/nonBL/table_kplr{star_id}.tailor-made.dat"

    #load in all the data and build the dictionaries
    data = {}
    times = []
    fluxes = []

    id_tracker = 0
    #should do splits as a uniform amount of time, not n splits per data set
    with open(filename) as f:

        #remove header
        for i in range(3):
            f.readline()

        #read in data
        for line in f:
            line = line.split()
            times.append(float(line[1]))
            fluxes.append(float(line[5]))

    times = np.array(times)
    fluxes = np.array(fluxes)

    print('data read in successfully')

    i = 0
    while i + len_segment < len(times):
        data[int(i/scan_shift)] = {
                   'times' : times[i:i+len_segment],
                   'fluxes' : fluxes[i:i+len_segment]
        }
        i += scan_shift

    print(len(data.keys()), 'scans' )

    times = np.arange(period-scan_n*scan_width/2, period+scan_n*scan_width/2, scan_width)

    print(f"calculating {len(times)} likelihoods for each scan...") #sanity check, should equal scan_n

    #the optimizer
    def likelihood(period, data, verbose_out=False):

        #scale the fluxes (not necesarry but it gives the "correct" scaling on the likelihood)
        fixed_fluxes = data['fluxes']
        fixed_fluxes = fixed_fluxes - min(fixed_fluxes) #doesn't change the relative errors
        fixed_fluxes = fixed_fluxes / max(fixed_fluxes)

        #fold the data based on the period
        folded_times = (data['times']/period) - (data['times']/period).astype(int)

        #shift the folded times so that the max value is always in the first spot
        tmp = fixed_fluxes.copy()
        tmp[np.isnan(tmp)] = 0
        max_ind = np.argmax(tmp)
        folded_times = folded_times - folded_times[max_ind]
        for i, el in enumerate(folded_times):
            if el < 0:
                folded_times[i] += 1

        #bin the data
        bin_width = 0.1
        overlap = 0.25 #fraction of bin width to overlap for the fit
        fit_binned_fluxes = []
        fit_binned_times = []
        binned_fluxes = []
        binned_times = []

        i = 0
        while i < 1:
            if i == 0: #have to wrap around for first and last bins
                indx = ((folded_times >= i-overlap*bin_width)*(folded_times < i+(1+overlap)*bin_width)) + (folded_times >= 1-overlap*bin_width)
                #wrap the data correctly
                tmp_folded_times = folded_times[indx]
                for j in range(len(tmp_folded_times)):
                    if tmp_folded_times[j] > 0.5:
                        tmp_folded_times[j] -= 1
            elif i >= 1-bin_width:
                indx = ((folded_times >= i-overlap*bin_width)*(folded_times < i+(1+overlap)*bin_width)) + (folded_times <= overlap*bin_width)
                #wrap the data correctly
                tmp_folded_times = folded_times[indx]
                for j in range(len(tmp_folded_times)):
                    if tmp_folded_times[j] < 0.5:
                        tmp_folded_times[j] += 1
            else:
                indx = (folded_times >= i-overlap*bin_width)*(folded_times < i+(1+overlap)*bin_width)
                tmp_folded_times = folded_times[indx]
            fit_binned_times.append(tmp_folded_times)
            fit_binned_fluxes.append(fixed_fluxes[indx])

            indx = (folded_times >= i)*(folded_times < i+bin_width)
            binned_fluxes.append(fixed_fluxes[indx])
            binned_times.append(folded_times[indx])

            i += bin_width
            i = round(i,2) #numerical error causes a bug otherwise

        #compute the dispersion of each bin w.r.t. a polynomial fit:    
        x2s = []
        for i in range(len(binned_times)):
            fit_indx = (np.logical_not(np.isnan(fit_binned_times[i])))*(np.logical_not(np.isnan(fit_binned_fluxes[i])))
            indx = (np.logical_not(np.isnan(binned_times[i])))*(np.logical_not(np.isnan(binned_fluxes[i])))
            if len(binned_fluxes[i][indx]) <= n_poly + 1: #can't compute a chi-squared for this
                x2s.append(np.nan)
            else:
                #poly fit
                popt = np.polyfit(fit_binned_times[i][fit_indx], fit_binned_fluxes[i][fit_indx], n_poly)
                y = np.polyval(popt, binned_times[i][indx])
                x2 = np.nansum((binned_fluxes[i][indx] - y)**2)
                x2 = x2 / (len(binned_fluxes[i][indx]) - (n_poly + 1))
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
    with open(f"{pickle_dir}best_periods_{star_id}_ls{len_segment}_ss{scan_shift}.pickle", 'wb') as p:
    	pickle.dump(best_periods, p)

    with open(f"{pickle_dir}all_like_list_{star_id}_ls{len_segment}_ss{scan_shift}.pickle", 'wb') as p:
    	pickle.dump(all_like_list, p)
