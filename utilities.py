import numpy as np
import matplotlib.pyplot as plt

""" Utilities functions """



def plot_insar_data_scatter(xout,yout,zout,image_name):
""" Scattered plot for the downsampled data """
	plt.figure()
    plt.scatter(xout,yout,zout,cmap='jet')
    plt.colorbar()
    #plt.clim()
    plt.savefig(image_name)



def plot_insar_sample(xout,yout,z,zout,xx1,xx2,yy1,yy2,image_name):
	plt.figure()
    plt.imshow(stacked,cmap='jet')
    plt.colorbar()
    #plt.clim()
    plt.savefig(image_name)

