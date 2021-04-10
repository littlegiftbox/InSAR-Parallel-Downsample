import os
import time
import numpy as np
import matplotlib.pyplot as plt

def main(): 

    # Read-in the data
    # Large dataset: Ridgecrest earthquake (Preparing 5000*5000 ish)
    # Smaller dataset: (1000~200 ish)

    # Downsample InSAR
    # xinsar, yinsar are vectors, while zinsar is a matrix
    Nmin = 3000 # minimum number of downsampled grids
    Nres_min = 20 # minimum grid size
    Nres_max = 400 # maxium grid size
    [xout,yout,zout,Npt,rms_out,xx1,xx2,yy1,yy2]=make_insar_downsample(xinsar,yinsar,zinsar,Nmin,Nres_min,Nres_max,method);
    
    # Downsample look vector

    # Plot the downsampled result

if __name__=='__main__':
    main()
