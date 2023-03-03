This folder contains the code and data that was used to create the figures from Donlon et al. (2023). 

The data in the /data folder was reduced on the MAPS v1.1 pipeline, with a total window size of 600 points and a scan length of 15 points. The original data can be found at https://konkoly.hu/KIK/data_en.html. We also utilized data from the NASA exoplanet archive, which is publicly available. 

There are three python notebooks included in this folder: 
 - `make_figures.inpyb` contains all the code required to make the figures from the paper
 - `mass_estimates.inpyb` contains code to calculate mass estimates for the companions
 - `make_modulation_diagrams.inpyb` contains code that constructs and plots the period and amplitude modulations for each star in the dataset
 
Some of the data that is used for the plotting routine/files that are hosted elsewhere/might be proprietary data that I don't want to re-host aren't included in this folder. You can download it yourself, or feel free to reach out to me for the files that you need. 

Additionally, if there's any other data/information that you would like from/related to the paper (i.e. the n-body simulations we ran, MCMC exoplanet fits, etc), you're welcome to reach out to me. I'll gladly work with you to make sure you get what you need.
