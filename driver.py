import os
import time
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
#from nonrecursive_downsample import *
from plotting_utilities import *
from nonrecursive_downsample import *
from make_insar_downsample import *
#from quad_insar_downsample_noncursive import *


def main(): 

    # Read-in the data: Ridgecrest earthquake 
    # Original dataset: matrix size 2330*3960 "Ridgecrest_ori.grd"
    # Smaller version: matrix size 1320*777 "Ridgecrest_small.grd"
    grid=nc.Dataset("Ridgecrest_small.grd")
    xinsar=np.array(grid['lon']) # 777
    yinsar=np.array(grid['lat']) # 1320
    zinsar=np.array(grid['z']) # 1320*777
    #print(np.shape(xinsar),np.shape(yinsar),np.shape(zinsar))
    #print(np.min(xinsar),np.max(xinsar))

    # Downsample InSAR
    # xinsar, yinsar are vectors, while zinsar is a matrix
    Nmin = 1000# 1000 # minimum number of downsampled grids
    Nres_min = 10#10 # minimum grid size
    Nres_max = 400 # maxium grid size
    method = 'mean'

    t0 = time.time()
    [xout, yout, zout, Npt, rms_out, xx1, xx2, yy1, yy2] = make_insar_downsample_nonrecursive(xinsar, yinsar, zinsar, Nmin, Nres_min,
                                                                                 Nres_max, method);
    print('Elapsed time: ', time.time() - t0)

    cmin = np.min(zout)
    cmax = np.max(zout)
    # plot_insar_data_scatter(xout,yout,zout,image_name);
    plot_insar_sample(xout, yout, zinsar, zout, xx1, xx2, yy1, yy2, cmin, cmax, 'test_nonrecursive.png');


    t0 = time.time()
    [xout, yout, zout, Npt, rms_out, xx1, xx2, yy1, yy2] = make_insar_downsample(xinsar, yinsar, zinsar, Nmin, Nres_min,
                                                                                 Nres_max, method);
    print('Elapsed time: ', time.time() - t0)

    cmin = np.min(zout)
    cmax = np.max(zout)
    # plot_insar_data_scatter(xout,yout,zout,image_name);
    plot_insar_sample(xout, yout, zinsar, zout, xx1, xx2, yy1, yy2, cmin, cmax, 'test_recursive.png');



if __name__=='__main__':
    main()
