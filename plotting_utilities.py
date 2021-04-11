import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle

""" Utilities functions """



def plot_insar_data_scatter(xout,yout,zout,image_name):
    """ Scattered plot for the downsampled data """
    plt.figure()
    plt.scatter(xout,yout,zout,cmap='jet')
    plt.colorbar()
    #plt.clim()
    plt.savefig(image_name)



def plot_insar_sample(xout,yout,z,zout,xx1,xx2,yy1,yy2,vmin,vmax,image_name):

    dx=xx2-xx1
    dy=yy2-yy1

    plt.figure(figsize=(8,8),dpi=600)
    color_boundary_object = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True);
    custom_cmap = cm.ScalarMappable(norm=color_boundary_object, cmap='rainbow');

    # Assemble Rectangular patches
    Npatch = np.size(zout)
    for i in range(0,Npatch):

        patch_color = custom_cmap.to_rgba(zout[i]); 
        patch_obj = Rectangle((xx1[i],yy1[i]),dx[i],dy[i],facecolor=patch_color,edgecolor='k',alpha=1.0)
        ax = plt.gca()
        ax.add_patch(patch_obj)


    custom_cmap.set_array(np.arange(vmin, vmax, 100));
    cb = plt.colorbar(custom_cmap);
    cb.set_label('put unit here', fontsize=12);
    plt.ylim([np.min(xx1), np.max(xx2)]);
    plt.xlim([np.min(yy1), np.max(yy2)]);
    plt.show()
    plt.savefig(image_name)


