import numpy as np

def make_insar_downsample(xinsar,yinsar,zinsar,Nmin,Nres_min,Nres_max,method):
    r1 = 10
    if method == 'mean':
        [xout, yout, zout, Npt, rms_out, xx1, xx2, yy1, yy2] = quad_decomp_mean(xinsar, yinsar, zinsar, r1, Nres_min,
                                                                                Nres_max, [], [], [], [], [], [], [],
                                                                                [], [])
        stop = 1

def quad_decomp_mean(xin,yin,zin,threshold, Nres_min,Nres_max,xout_in,yout_in,zout_in, Ndata_in,rms_in,xx1_in,xx2_in,yy1_in,yy2_in):
    nx = np.shape(xin)[0]
    ny = np.shape(yin)[0]
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
    if ((nx <= Nres_min) | (ny<=Nres_min)):
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
    else:
        nx1 = 0
        nx2 = int(np.floor(nx / 2))
        nx3 = nx2
        nx4 = nx - 1

        ny1 = 0
        ny2 = int(np.floor(ny / 2))
        ny3 = ny2
        ny4 = ny - 1

        x1 = xin[nx1:nx2 + 1]
        x2 = xin[nx3:nx4 + 1]
        x3 = x1
        x4 = x2

        y1 = yin[ny1:ny2 + 1]
        y2 = y1

        y3 = yin[ny3:ny4 + 1]
        y4 = y3

        z1 = zin[ny1:ny2 + 1, nx1: nx2 + 1]
        z2 = zin[ny1:ny2 + 1, nx3: nx4 + 1]
        z3 = zin[ny3:ny4 + 1, nx1: nx2 + 1]
        z4 = zin[ny3:ny4 + 1, nx3: nx4 + 1]

        xmin1 = xin[nx1]
        xmax1 = xin[nx2]
        xmin2 = xin[nx3]
        xmax2 = xin[nx4]

        ymin1 = yin[ny1]
        ymax1 = yin[ny2]
        ymin2 = yin[ny3]
        ymax2 = yin[ny4]

        [rms1, N1, r1_good, x1_out, y1_out, z1_out] = rms_block_demean(x1, y1, z1, Nres_min, Nres_max)
        [rms2, N2, r2_good, x2_out, y2_out, z2_out] = rms_block_demean(x2, y2, z2, Nres_min, Nres_max)
        [rms3, N3, r3_good, x3_out, y3_out, z3_out] = rms_block_demean(x3, y3, z3, Nres_min, Nres_max)
        [rms4, N4, r4_good, x4_out, y4_out, z4_out] = rms_block_demean(x4, y4, z4, Nres_min, Nres_max)

        if ((rms1 <= threshold) & (N1 > 0) & (r1_good > r_good_default)):
            xout.append(x1_out)
            yout.append(y1_out)
            zout.append(z1_out)

            zgood_this_block = z1[~np.isnan(z1)]
            Ngood_this_block = np.shape(zgood_this_block)[0]
            dz_this_block = zgood_this_block - z1_out
            rms_this_block = np.sqrt(sum(dz_this_block** 2) / Ngood_this_block)
            if (rms_this_block < 1.0e-6):
                rms_this_block = rms_default

            rms_out.append(rms_this_block)
            Ndata.append(Ngood_this_block)
            xx1_this_block = xmin1
            xx2_this_block = xmax1
            yy1_this_block = ymin1
            yy2_this_block = ymax1

            xx1.append(xx1_this_block)
            yy1.append(yy1_this_block)
            xx2.append(xx2_this_block)
            yy2.append(yy2_this_block)
        elif ((rms1>threshold) & (N1>0)):
            xout_in = xout
            yout_in = yout
            zout_in = zout
            xx1_in = xx1
            xx2_in = xx2
            yy1_in = yy1
            yy2_in = yy2
            Ndata_in = Ndata
            rms_in = rms_out
            [xout, yout, zout, Ndata, rms_out, xx1, xx2, yy1, yy2] = quad_decomp_mean(x1, y1, z1,
                                                                                      threshold, Nres_min, Nres_max,
                                                                                      xout_in, yout_in, zout_in,
                                                                                      Ndata_in, rms_in, xx1_in, xx2_in,
                                                                                      yy1_in, yy2_in);

        if ((rms2 <= threshold) & (N2 > 0) & (r2_good > r_good_default)):
            xout.append(x2_out)
            yout.append(y2_out)
            zout.append(z2_out)

            zgood_this_block = z2[~np.isnan(z2)]
            Ngood_this_block = np.shape(zgood_this_block)[0]
            dz_this_block = zgood_this_block - z2_out
            rms_this_block = np.sqrt(sum(dz_this_block** 2) / Ngood_this_block)
            if (rms_this_block < 1.0e-6):
                rms_this_block = rms_default;

            rms_out.append(rms_this_block)
            Ndata.append(Ngood_this_block)
            xx1_this_block = xmin2
            xx2_this_block = xmax2
            yy1_this_block = ymin1
            yy2_this_block = ymax1

            xx1.append(xx1_this_block)
            yy1.append(yy1_this_block)
            xx2.append(xx2_this_block)
            yy2.append(yy2_this_block)
        elif ((rms2>threshold) & (N2>0)):
            xout_in = xout
            yout_in = yout
            zout_in = zout
            xx1_in = xx1
            xx2_in = xx2
            yy1_in = yy1
            yy2_in = yy2
            Ndata_in = Ndata
            rms_in = rms_out
            [xout, yout, zout, Ndata, rms_out, xx1, xx2, yy1, yy2] = quad_decomp_mean(x2,y2,z2,threshold,
                                                                                      Nres_min,Nres_max,xout_in,yout_in,
                                                                                      zout_in,Ndata_in,rms_in,
                                                                                      xx1_in,xx2_in,yy1_in,yy2_in)

        if ((rms3 <= threshold) & (N3 > 0) & (r3_good > r_good_default)):
            xout.append(x3_out)
            yout.append(y3_out)
            zout.append(z3_out)

            zgood_this_block = z3[~np.isnan(z3)]
            Ngood_this_block = np.shape(zgood_this_block)[0]
            dz_this_block = zgood_this_block - z3_out
            rms_this_block = np.sqrt(sum(dz_this_block** 2) / Ngood_this_block)
            if (rms_this_block < 1.0e-6):
                rms_this_block = rms_default

            rms_out.append(rms_this_block)
            Ndata.append( Ngood_this_block)
            xx1_this_block = xmin1
            xx2_this_block = xmax1
            yy1_this_block = ymin2
            yy2_this_block = ymax2

            xx1.append(xx1_this_block)
            yy1.append(yy1_this_block)
            xx2.append(xx2_this_block)
            yy2.append(yy2_this_block)
        elif ((rms3>threshold) & (N3>0)):
            xout_in = xout
            yout_in = yout
            zout_in = zout
            xx1_in = xx1
            xx2_in = xx2
            yy1_in = yy1
            yy2_in = yy2
            Ndata_in = Ndata
            rms_in = rms_out
            [xout, yout, zout, Ndata, rms_out, xx1, xx2, yy1, yy2] = quad_decomp_mean(x3,y3,z3,threshold,
                                                                                      Nres_min,Nres_max,xout_in,yout_in,
                                                                                      zout_in,Ndata_in,rms_in,
                                                                                      xx1_in,xx2_in,yy1_in,yy2_in)

        if ((rms4 <= threshold) & (N4 > 0) & (r4_good > r_good_default)):
            xout.append(x4_out)
            yout.append(y4_out)
            zout.append(z4_out)

            zgood_this_block = z4[~np.isnan(z4)]
            Ngood_this_block = np.shape(zgood_this_block)[0]
            dz_this_block = zgood_this_block - z4_out
            rms_this_block = np.sqrt(sum(dz_this_block** 2) / Ngood_this_block)
            if (rms_this_block < 1.0e-6):
                rms_this_block = rms_default

            rms_out.append(rms_this_block)
            Ndata.append(Ngood_this_block)
            xx1_this_block = xmin2
            xx2_this_block = xmax2
            yy1_this_block = ymin2
            yy2_this_block = ymax2

            xx1.append(xx1_this_block)
            yy1.append(yy1_this_block)
            xx2.append(xx2_this_block)
            yy2.append(yy2_this_block)
        elif ((rms4>threshold) & (N4>0)):
            xout_in = xout
            yout_in = yout
            zout_in = zout
            xx1_in = xx1
            xx2_in = xx2
            yy1_in = yy1
            yy2_in = yy2
            Ndata_in = Ndata
            rms_in = rms_out
            [xout, yout, zout, Ndata, rms_out, xx1, xx2, yy1, yy2] = quad_decomp_mean(x4,y4,z4,threshold,
                                                                                      Nres_min,Nres_max,xout_in,yout_in,
                                                                                      zout_in,Ndata_in,rms_in,
                                                                                      xx1_in,xx2_in,yy1_in,yy2_in)

    return xout,yout,zout,Ndata,rms_out,xx1,xx2,yy1,yy2




def rms_block_demean(x,y,z,Nres_min,Nres_max):
    [xx, yy] = np.meshgrid(x,y)
    indx_good = ~np.isnan(z)
    [nx, ny] = np.shape(z)
    n_block = nx*ny
    xdata = xx[indx_good]
    ydata = yy[indx_good]
    zdata = z[indx_good]

    Ngood = np.shape(zdata)[0]
    r_good = float(Ngood)/n_block
    if (Ngood > 0):
        xout = np.mean(xdata)
        yout = np.mean(ydata)
        zout = np.mean(zdata)
        lx = np.shape(np.unique(x))[0]
        ly = np.shape(np.unique(y))[0]
        if ((Ngood<=3) | (lx<=Nres_min) | (ly<=Nres_min)):
            rms_out = 0
        elif ((Ngood > 5) & ((lx>2) & (lx < Nres_max)) & ((ly>2) & (ly < Nres_max))):
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




