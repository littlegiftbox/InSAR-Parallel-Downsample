import numpy as np

def quad_decomp_mean(xin,yin,zin,threshold, Nres_min,Nres_max,xout_in,yout_in,zout_in, Ndata_in,rms_in,xx1_in,xx2_in,yy1_in,yy2_in):
    nx = np.size(xin)
    ny = np.size(yin)
    xout = xout_in
    yout = yout_in
    zout = zout_in
    Ndata = Ndata_in
    rms_out = rms_in
    xx1 = xx1_in
    xx2 = xx2_in
    yy1 = yy1_in
    yy2 = yy2_in

    rms_default = 10
    r_good_default = 0.2
    if (nx <= Nres_min | ny<=Nres_min):
        xout_block_total = np.mean(xin)
        yout_block_total = np.mean(yin)
        z_block_good = zin[~np.isnan(zin)]
        N_block_good = np.size(z_block_good)
        n_block = nx * ny
        r_good = N_block_good / n_block
        if (N_block_good > 0 & r_good > r_good_default):
            zout_block_total = np.mean(z_block_good)
            dz_block = z_block_good - zout_block_total
            rms_block_total = np.sqrt(sum(sum(dz_block** 2)) / N_block_good)
            if (rms_block_total < 1.0e-6):
                rms_block_total = 10
            xout = [xout, xout_block_total]
            yout = [yout, yout_block_total]
            zout = [zout, zout_block_total]
            Ndata = [Ndata, N_block_good]
            rms_out = [rms_out, rms_block_total]
            xx1 = [xx1, xin(1)]
            xx2 = [xx2, xin(nx)]
            yy1 = [yy1, yin(1)]
            yy2 = [yy2, yin(ny)]

def rms_block_demean(x,y,z,Nres_min,Nres_max):
    [xx, yy] = np.meshgrid(x,y)
    indx_good = z != np.nan
    [nx, ny] = np.shape(z)
    n_block = nx*ny
    xdata = xx[indx_good]
    ydata = yy[indx_good]
    zdata = z[indx_good]

    Ngood = np.shape(zdata)
    r_good = float(Ngood/n_block)
    if (Ngood > 0):
        xout = np.mean(xdata)
        yout = np.mean(ydata)
        zout = np.mean(zdata)
        lx = np.shape(np.unique(x))
        ly = np.shape(np.unique(x))
        if ((Ngood<=3) | (lx<=Nres_min) | (ly<=Nres_min)):
            rms_out = 0
        elif ((Ngood > 5) & (lx>2 & lx < Nres_max) & (ly>2 & ly < Nres_max)):
            zz = zdata
            zzfit = np.mean(zz)
            dz = zz-zzfit
            rms_out = np.sqrt(np.sum(dz**2)/Ngood)
        else:
            rms_out = 1000
    else:
        xout = np.nan
        yout = np.nan
        zout = np.nan
        rms_out = 0
    return rms_out, Ngood, r_good, xout, yout, zout



if __name__ == '__main__':
    x = np.zeros((1, 389)) + 1
    y = np.zeros((1, 661)) + 2
    z = np.zeros((661, 389)) + 3
    Nres_min = 20
    Nres_max = 400
    rms_block_demean(x, y, z, Nres_min, Nres_max)




