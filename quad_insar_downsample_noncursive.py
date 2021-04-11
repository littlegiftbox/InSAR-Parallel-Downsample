import numpy as np
from make_insar_downsample import *

class node:
    def __init__(self, minx, miny, maxx, maxy):
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.isnode = True
        self.isleaf = False
        self.rms = 0
        self.N = 0
        self.xout = 0
        self.yout = 0
        self.zout = 0

def make_insar_downsample_nonrecursive(xinsar,yinsar,zinsar,Nmin,Nres_min,Nres_max,method):
    r1 = 10
    if method == 'mean':
        [xout, yout, zout, Npt, rms_out, xx1, xx2, yy1, yy2] = quad_decomp_mean_nonrecursive(xinsar, yinsar, zinsar, r1, Nres_min,
                                                                                Nres_max, [], [], [], [], [], [], [],
                                                                                [], [])
    Ndata = len(zout)
    Nint = 0
    while (Ndata < Nmin):
        N1 = len(zout)
        r1 = r1 * 0.85
        if method == 'mean':
            [xout, yout, zout, Npt, rms_out, xx1, xx2, yy1, yy2] = quad_decomp_mean_nonrecursive(xinsar, yinsar, zinsar, r1,
                                                                                    Nres_min, Nres_max, [], [], [], [],
                                                                                    [], [], [], [], [])
        Ndata = len(zout)
        N2 = len(zout)
        Nint = Nint + 1
        if ((N2 > 0.8 * Nmin) & ((N2-N1) < 0.005 * N1)):
            break
    xout = np.array(xout)
    yout = np.array(yout)
    zout = np.array(zout)
    Npt = np.array(Npt)
    rms_out = np.array(rms_out)
    xx1 = np.array(xx1)
    xx2 = np.array(xx2)
    yy1 = np.array(yy1)
    yy2 = np.array(yy2)
    return xout, yout, zout, Npt, rms_out, xx1, xx2, yy1, yy2


def quad_decomp_mean_nonrecursive(xin,yin,zin,threshold, Nres_min,Nres_max,xout_in,yout_in,zout_in,
                                  Ndata_in,rms_in,xx1_in,xx2_in,yy1_in,yy2_in):
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

    max_depth = int(np.floor(np.log2(nx/threshold))+1)
    # create array aseemble for each depth featuring a size of 4^(depth)
    index_array_assemble = []
    for i_depth in range(max_depth):
        index_array_assemble.append([])

    # construct the root node
    this_node = node(0, 0, nx, ny)
    index_array_assemble[0].append(this_node)

    for i_depth in range(max_depth-1):
        num_nodes = np.size(index_array_assemble[i_depth])
        #print('i_depth:{}'.format(i_depth))
        #print('number of nodes: {}'.format(num_nodes))
        total_num_check = 0
        for i_node in range(num_nodes):
            #print('i_node:{}'.format(i_node))
            if index_array_assemble[i_depth][i_node].isnode:
                if index_array_assemble[i_depth][i_node].isleaf == 0:
                    nx1 = index_array_assemble[i_depth][i_node].minx
                    xincrement = int((index_array_assemble[i_depth][i_node].maxx -
                                      index_array_assemble[i_depth][i_node].minx)/2)
                    nx2 = nx1 + xincrement
                    nx3 = nx2
                    nx4 = index_array_assemble[i_depth][i_node].maxx

                    ny1 = index_array_assemble[i_depth][i_node].miny
                    yincrement = int((index_array_assemble[i_depth][i_node].maxy -
                                      index_array_assemble[i_depth][i_node].miny)/2)
                    ny2 = ny1 + yincrement
                    ny3 = ny2
                    ny4 = index_array_assemble[i_depth][i_node].maxy

                    index_array_assemble[i_depth+1].append(node(nx1, ny1, nx2, ny2))
                    index_array_assemble[i_depth+1].append(node(nx3, ny1, nx4, ny2))
                    index_array_assemble[i_depth+1].append(node(nx1, ny3, nx2, ny4))
                    index_array_assemble[i_depth+1].append(node(nx3, ny3, nx4, ny4))

                    for j_index in range(1, 5):
                        this_index = len(index_array_assemble[i_depth+1])-j_index
                        this_z = zin[index_array_assemble[i_depth+1][this_index].miny:index_array_assemble[i_depth+1][this_index].maxy,
                                 index_array_assemble[i_depth+1][this_index].minx:index_array_assemble[i_depth+1][this_index].maxx]
                        this_x = xin[index_array_assemble[i_depth+1][this_index].minx:index_array_assemble[i_depth+1][this_index].maxx]
                        this_y = yin[index_array_assemble[i_depth+1][this_index].miny:index_array_assemble[i_depth+1][this_index].maxy]
                        [rms, N, r_good, x_out, y_out, z_out] = rms_block_demean(this_x, this_y, this_z, Nres_min, Nres_max)
                        nx = len(this_x)
                        ny = len(this_y)
                        leaf_crit1 = ((rms <= threshold) & (N > 0) & (r_good > r_good_default))
                        leaf_crit2 = ((nx <= Nres_min) | (ny<=Nres_min))
                        total_num_check = total_num_check + np.size(this_z)
                        #print(np.size(this_z))
                        if (leaf_crit1 | leaf_crit2):
                            zgood_this_block = this_z[~np.isnan(this_z)]
                            Ngood_this_block = np.size(zgood_this_block)
                            dz_this_block = zgood_this_block - z_out
                            if ((Ngood_this_block > 0) & (r_good > r_good_default)):
                                rms_this_block = np.sqrt(sum(dz_this_block ** 2) / Ngood_this_block)
                                if (rms_this_block < 1.0e-6):
                                    rms_this_block = rms_default
                                index_array_assemble[i_depth + 1][this_index].isleaf = True
                                index_array_assemble[i_depth + 1][this_index].rms = rms_this_block
                                index_array_assemble[i_depth + 1][this_index].N = Ngood_this_block
                                index_array_assemble[i_depth + 1][this_index].x_out = x_out
                                index_array_assemble[i_depth + 1][this_index].y_out = y_out
                                index_array_assemble[i_depth + 1][this_index].z_out = z_out
                            else:
                                #print(r_good)
                                index_array_assemble[i_depth + 1][this_index].isnode = False
        #print('total num : {}'.format(total_num_check))


    total_num_check = 0
    for i_depth in range(max_depth):
        num_nodes = np.size(index_array_assemble[i_depth])
        for i_node in range(num_nodes):
            if index_array_assemble[i_depth][i_node].isleaf:
                #print('this depth is {}, this node is {}'.format(i_depth, i_node))
                xout.append(index_array_assemble[i_depth][i_node].x_out)
                yout.append(index_array_assemble[i_depth][i_node].y_out)
                zout.append(index_array_assemble[i_depth][i_node].z_out)
                rms_out.append(index_array_assemble[i_depth][i_node].rms)
                Ndata.append(index_array_assemble[i_depth][i_node].N)
                xx1.append(xin[index_array_assemble[i_depth][i_node].minx])
                xx2.append(xin[index_array_assemble[i_depth][i_node].maxx-1])
                yy1.append(yin[index_array_assemble[i_depth][i_node].miny])
                yy2.append(yin[index_array_assemble[i_depth][i_node].maxy-1])
                total_num_check += index_array_assemble[i_depth][i_node].N
    return xout, yout, zout, Ndata, rms_out, xx1, xx2, yy1, yy2


