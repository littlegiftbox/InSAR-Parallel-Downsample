import numpy as np
from make_insar_downsample import *
import math
from numba import cuda
import cmath

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
        [xout, yout, zout, Npt, rms_out, xx1, xx2, yy1, yy2] = quad_decomp_mean_serial_noclass(xinsar, yinsar, zinsar, r1, Nres_min,
                                                                                Nres_max, [], [], [], [], [], [], [],
                                                                                [], [])
    Ndata = len(zout)
    Nint = 0
    while (Ndata < Nmin):
        N1 = len(zout)
        r1 = r1 * 0.85
        if method == 'mean':
            [xout, yout, zout, Npt, rms_out, xx1, xx2, yy1, yy2] = quad_decomp_mean_serial_noclass(xinsar, yinsar, zinsar, r1,
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

    max_depth = int(math.log((nx*ny)/(Nres_min**2), 4))+2
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


def quad_decomp_mean_serial(xin,yin,zin,threshold, Nres_min,Nres_max,xout_in,yout_in,zout_in,
                                  Ndata_in,rms_in,xx1_in,xx2_in,yy1_in,yy2_in):
    nx = len(xin)
    ny = len(yin)
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
    max_depth = int(math.log((nx*ny)/(Nres_min**2), 4))+2
    node_total = int((nx*ny)/(Nres_min**2))+1
    # create array aseemble for each depth featuring a size of 4^(depth)
    index_array_assemble = np.empty(node_total, dtype=node)
    index_array_assemble[0] = node(0, 0, nx, ny)
    psindx = 0
    peindx = 1
    csindx = 0
    for i_depth in range(max_depth-1):
        #print('i_depth:{}'.format(i_depth))
        #print('number of nodes: {}'.format(num_nodes))
        this_layer_node = 0
        for i_node in range(psindx, peindx):
            #print('i_node:{}'.format(i_node))
            if index_array_assemble[i_node].isnode:
                if index_array_assemble[i_node].isleaf == 0:
                    nx1 = index_array_assemble[i_node].minx
                    xincrement = int((index_array_assemble[i_node].maxx -
                                      index_array_assemble[i_node].minx)/2)
                    nx2 = nx1 + xincrement
                    nx3 = nx2
                    nx4 = index_array_assemble[i_node].maxx

                    ny1 = index_array_assemble[i_node].miny
                    yincrement = int((index_array_assemble[i_node].maxy -
                                      index_array_assemble[i_node].miny)/2)
                    ny2 = ny1 + yincrement
                    ny3 = ny2
                    ny4 = index_array_assemble[i_node].maxy

                    index_array_assemble[csindx+1] = node(nx1, ny1, nx2, ny2)
                    index_array_assemble[csindx+2] = node(nx3, ny1, nx4, ny2)
                    index_array_assemble[csindx+3] = node(nx1, ny3, nx2, ny4)
                    index_array_assemble[csindx+4] = node(nx3, ny3, nx4, ny4)

                    csindx += 4
                    this_layer_node += 4
                    for j_index in range(4):
                        this_index = csindx-j_index
                        this_z = zin[index_array_assemble[this_index].miny:index_array_assemble[this_index].maxy,
                                 index_array_assemble[this_index].minx:index_array_assemble[this_index].maxx]
                        this_x = xin[index_array_assemble[this_index].minx:index_array_assemble[this_index].maxx]
                        this_y = yin[index_array_assemble[this_index].miny:index_array_assemble[this_index].maxy]
                        [rms, N, r_good, x_out, y_out, z_out] = rms_block_demean(this_x, this_y, this_z, Nres_min, Nres_max)
                        nx = len(this_x)
                        ny = len(this_y)
                        leaf_crit1 = ((rms <= threshold) & (N > 0) & (r_good > r_good_default))
                        leaf_crit2 = ((nx <= Nres_min) | (ny<=Nres_min))
                        #print(np.size(this_z))
                        if (leaf_crit1 | leaf_crit2):
                            zgood_this_block = this_z[~np.isnan(this_z)]
                            Ngood_this_block = len(zgood_this_block)
                            dz_this_block = zgood_this_block - z_out
                            if ((Ngood_this_block > 0) & (r_good > r_good_default)):
                                rms_this_block = math.sqrt(sum(dz_this_block ** 2) / Ngood_this_block)
                                if (rms_this_block < 1.0e-6):
                                    rms_this_block = rms_default
                                index_array_assemble[this_index].isleaf = True
                                index_array_assemble[this_index].rms = rms_this_block
                                index_array_assemble[this_index].N = Ngood_this_block
                                index_array_assemble[this_index].x_out = x_out
                                index_array_assemble[this_index].y_out = y_out
                                index_array_assemble[this_index].z_out = z_out
                            else:
                                #print(r_good)
                                index_array_assemble[this_index].isnode = False

        psindx = peindx
        peindx = peindx + this_layer_node
        #print('total num : {}'.format(total_num_check))
    total_num_check = 0
    for i_node in range(node_total):
        if index_array_assemble[i_node]:
            if index_array_assemble[i_node].isleaf:
                #print('this depth is {}, this node is {}'.format(i_depth, i_node))
                xout.append(index_array_assemble[i_node].x_out)
                yout.append(index_array_assemble[i_node].y_out)
                zout.append(index_array_assemble[i_node].z_out)
                rms_out.append(index_array_assemble[i_node].rms)
                Ndata.append(index_array_assemble[i_node].N)
                xx1.append(xin[index_array_assemble[i_node].minx])
                xx2.append(xin[index_array_assemble[i_node].maxx-1])
                yy1.append(yin[index_array_assemble[i_node].miny])
                yy2.append(yin[index_array_assemble[i_node].maxy-1])
                total_num_check += index_array_assemble[i_node].N
        else:
            break
    return xout, yout, zout, Ndata, rms_out, xx1, xx2, yy1, yy2


def quad_decomp_mean_serial_noclass(xin,yin,zin,zmaskin,threshold, Nres_min,Nres_max,xout_in,yout_in,zout_in,
                                  Ndata_in,rms_in,xx1_in,xx2_in,yy1_in,yy2_in):
    nx = len(xin)
    ny = len(yin)
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
    max_depth = int(math.log((nx*ny)/(Nres_min**2), 4))+2
    node_total = int((nx*ny)/(Nres_min**2))+1
    # create array aseemble for each depth featuring a size of 4^(depth)
    minx = np.empty(node_total, dtype=int)
    maxx = np.empty(node_total, dtype=int)
    miny = np.empty(node_total, dtype=int)
    maxy = np.empty(node_total, dtype=int)
    isnode = np.empty(node_total, dtype=bool)
    isnode[:] = False
    isleaf = np.empty(node_total, dtype=bool)
    isleaf[:] = False

    minx[0] = 0
    maxx[0] = nx
    miny[0] = 0
    maxy[0] = ny
    isnode[0] = True
    isleaf[0] = False

    psindx = 0
    peindx = 1
    csindx = 0
    for i_depth in range(max_depth-1):
        this_layer_node = 0
        for i_node in range(psindx, peindx):
            if isnode[i_node]:
                if isleaf[i_node] == 0:
                    nx1 = minx[i_node]
                    xincrement = int((maxx[i_node] -
                                      minx[i_node])/2)
                    nx2 = nx1 + xincrement
                    nx3 = nx2
                    nx4 = maxx[i_node]

                    ny1 = miny[i_node]
                    yincrement = int((maxy[i_node] - miny[i_node])/2)
                    ny2 = ny1 + yincrement
                    ny3 = ny2
                    ny4 = maxy[i_node]

                    minx[csindx+1:csindx+5] = [nx1, nx3, nx1, nx3]
                    maxx[csindx+1:csindx+5] = [nx2, nx4, nx2, nx4]
                    miny[csindx+1:csindx+5] = [ny1, ny1, ny3, ny3]
                    maxy[csindx+1:csindx+5] = [ny2, ny2, ny4, ny4]

                    csindx += 4
                    this_layer_node += 4
                    for j_index in range(4):
                        this_index = csindx-j_index
                        this_z = zin[miny[this_index]:maxy[this_index],
                                 minx[this_index]:maxx[this_index]]
                        this_x = xin[minx[this_index]:maxx[this_index]]
                        this_y = yin[miny[this_index]:maxy[this_index]]
                        [rms, N, r_good] = rms_block_demean_serial(this_x, this_y, this_z, Nres_min, Nres_max)
                        nx = len(this_x)
                        ny = len(this_y)
                        leaf_crit1 = ((rms <= threshold) & (N > 0) & (r_good > r_good_default))
                        leaf_crit2 = ((nx <= Nres_min) | (ny<=Nres_min))
                        #print(np.size(this_z))
                        if (leaf_crit1 | leaf_crit2):
                            zgood_this_block = this_z[~np.isnan(this_z)]
                            Ngood_this_block = len(zgood_this_block)
                            dz_this_block = zgood_this_block - sum(zgood_this_block)/len(zgood_this_block)
                            if ((Ngood_this_block > 0) & (r_good > r_good_default)):
                                isleaf[this_index] = True
                                isnode[this_index] = True
                            else:
                                #print(r_good)
                                isnode[this_index] = False
                                isleaf[this_index] = False
                        else:
                            isnode[this_index] = True
                            isleaf[this_index] = False
        psindx = peindx
        peindx = peindx + this_layer_node

    for i_node in range(node_total):
        if (isleaf[i_node] & isnode[i_node]):
            #print('this depth is {}, this node is {}'.format(i_depth, i_node))
            this_z = zin[miny[i_node]:maxy[i_node],
                     minx[i_node]:maxx[i_node]]
            this_x = xin[minx[i_node]:maxx[i_node]]
            this_y = yin[miny[i_node]:maxy[i_node]]
            z_block_good = this_z[~np.isnan(this_z)]
            N_block_good = len(z_block_good)
            xout_block_total = sum(this_x)/len(this_x)
            yout_block_total = sum(this_y)/len(this_y)
            n_block = nx * ny
            r_good = N_block_good / n_block
            zout_block_total = sum(z_block_good)/len(z_block_good)
            dz_block = z_block_good - zout_block_total
            rms_block_total = np.sqrt((sum(dz_block ** 2) / N_block_good))
            if (rms_block_total < 1.0e-6):
                rms_block_total = 10
            xout.append(xout_block_total)
            yout.append(yout_block_total)
            zout.append(zout_block_total)
            rms_out.append(rms_block_total)
            Ndata.append(N_block_good)
            xx1.append(xin[minx[i_node]])
            xx2.append(xin[maxx[i_node]-1])
            yy1.append(yin[miny[i_node]])
            yy2.append(yin[maxy[i_node]-1])

    return xout, yout, zout, Ndata, rms_out, xx1, xx2, yy1, yy2

def quad_decomp_mean_parallel(xin,yin,zin,threshold, Nres_min,Nres_max,xout_in,yout_in,zout_in,
                                  Ndata_in,rms_in,xx1_in,xx2_in,yy1_in,yy2_in):
    nx = len(xin)
    ny = len(yin)
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
    max_depth = int(math.log((nx*ny)/(Nres_min**2), 4))+2
    node_total = int((nx*ny)/(Nres_min**2))+1
    # create array aseemble for each depth featuring a size of 4^(depth)
    minx = np.empty(node_total, dtype=int)
    maxx = np.empty(node_total, dtype=int)
    miny = np.empty(node_total, dtype=int)
    maxy = np.empty(node_total, dtype=int)
    isnode = np.empty(node_total, dtype=bool)
    isnode[:] = False
    isleaf = np.empty(node_total, dtype=bool)
    isleaf[:] = False

    minx[0] = 0
    maxx[0] = nx
    miny[0] = 0
    maxy[0] = ny
    isnode[0] = True
    isleaf[0] = False

    psindx = 0
    peindx = 1
    csindx = 0


    for i_depth in range(max_depth-1):
        this_layer_node = 0
        decomp_one_depth[2496, 1](minx, miny, maxx, maxy, isnode, isleaf, xin, yin, zin, csindx, threshold,
                                  r_good_default, this_layer_node, psindx, peindx)
        psindx = peindx
        peindx = peindx + this_layer_node

    for i_node in range(node_total):
        if (isleaf[i_node] & isnode[i_node]):
            #print('this depth is {}, this node is {}'.format(i_depth, i_node))
            this_z = zin[miny[i_node]:maxy[i_node],
                     minx[i_node]:maxx[i_node]]
            this_x = xin[minx[i_node]:maxx[i_node]]
            this_y = yin[miny[i_node]:maxy[i_node]]
            z_block_good = this_z[~np.isnan(this_z)]
            N_block_good = len(z_block_good)
            xout_block_total = sum(this_x)/len(this_x)
            yout_block_total = sum(this_y)/len(this_y)
            n_block = nx * ny
            r_good = N_block_good / n_block
            zout_block_total = sum(z_block_good)/len(z_block_good)
            dz_block = z_block_good - zout_block_total
            rms_block_total = np.sqrt((sum(dz_block ** 2) / N_block_good))
            if (rms_block_total < 1.0e-6):
                rms_block_total = 10
            xout.append(xout_block_total)
            yout.append(yout_block_total)
            zout.append(zout_block_total)
            rms_out.append(rms_block_total)
            Ndata.append(N_block_good)
            xx1.append(xin[minx[i_node]])
            xx2.append(xin[maxx[i_node]-1])
            yy1.append(yin[miny[i_node]])
            yy2.append(yin[maxy[i_node]-1])

    return xout, yout, zout, Ndata, rms_out, xx1, xx2, yy1, yy2


@cuda.jit(device=True)
def decomp_one_depth(minx, miny, maxx, maxy, isnode, isleaf, xin, yin, zin, csindx, threshold, r_good_default,
                     this_layer_node, psindx, peindx):
    pos = cuda.grid(1)
    if pos in range(peindx - psindx):
        i_node = pos + psindx
        if isnode[i_node]:
            if isleaf[i_node] == 0:
                nx1 = minx[i_node]
                xincrement = int((maxx[i_node] -
                                  minx[i_node]) / 2)
                nx2 = nx1 + xincrement
                nx3 = nx2
                nx4 = maxx[i_node]

                ny1 = miny[i_node]
                yincrement = int((maxy[i_node] - miny[i_node]) / 2)
                ny2 = ny1 + yincrement
                ny3 = ny2
                ny4 = maxy[i_node]

                minx[csindx + 1:csindx + 5] = [nx1, nx3, nx1, nx3]
                maxx[csindx + 1:csindx + 5] = [nx2, nx4, nx2, nx4]
                miny[csindx + 1:csindx + 5] = [ny1, ny1, ny3, ny3]
                maxy[csindx + 1:csindx + 5] = [ny2, ny2, ny4, ny4]

                csindx += 4
                this_layer_node += 4
                for j_index in range(4):
                    this_index = csindx - j_index
                    this_z = zin[miny[this_index]:maxy[this_index],
                             minx[this_index]:maxx[this_index]]
                    this_x = xin[minx[this_index]:maxx[this_index]]
                    this_y = yin[miny[this_index]:maxy[this_index]]
                    [rms, N, r_good] = rms_block_demean_sim(this_x, this_y, this_z, Nres_min, Nres_max)
                    nx = len(this_x)
                    ny = len(this_y)
                    leaf_crit1 = ((rms <= threshold) & (N > 0) & (r_good > r_good_default))
                    leaf_crit2 = ((nx <= Nres_min) | (ny <= Nres_min))
                    # print(np.size(this_z))
                    if (leaf_crit1 | leaf_crit2):
                        zgood_this_block = this_z[~np.isnan(this_z)]
                        Ngood_this_block = len(zgood_this_block)
                        dz_this_block = zgood_this_block - sum(zgood_this_block) / len(zgood_this_block)
                        if ((Ngood_this_block > 0) & (r_good > r_good_default)):
                            isleaf[this_index] = True
                            isnode[this_index] = True
                        else:
                            # print(r_good)
                            isnode[this_index] = False
                            isleaf[this_index] = False
                    else:
                        isnode[this_index] = True
                        isleaf[this_index] = False


@cuda.jit(device=True)
def rms_block_demean_sim(x,y,z,Nres_min,Nres_max):
    indx_good = ~np.isnan(z)
    [nx, ny] = np.shape(z)
    n_block = nx*ny
    zdata = z[indx_good]
    Ngood = np.shape(zdata)[0]
    r_good = float(Ngood)/n_block
    if (Ngood > 0):
        lx = np.shape(np.unique(x))[0]
        ly = np.shape(np.unique(y))[0]
        if ((Ngood<=3) | (lx<=Nres_min) | (ly<=Nres_min)):
            rms_out = 0
        elif ((Ngood > 5) & ((lx>2) & (lx < Nres_max)) & ((ly>2) & (ly < Nres_max))):
            zz = zdata
            zzfit = sum(zz)/len(zz)
            dz = zz-zzfit
            rms_out = np.sqrt(np.sum(dz**2)/Ngood)
        else:
            rms_out = 1000
    else:
        rms_out = 0
    return rms_out, Ngood, r_good


def rms_block_demean_serial(x,y,z,Nres_min,Nres_max):
    indx_good = ~np.isnan(z)
    [nx, ny] = np.shape(z)
    n_block = nx*ny
    zdata = z[indx_good]
    Ngood = np.shape(zdata)[0]
    r_good = float(Ngood)/n_block
    if (Ngood > 0):
        lx = np.shape(np.unique(x))[0]
        ly = np.shape(np.unique(y))[0]
        if ((Ngood<=3) | (lx<=Nres_min) | (ly<=Nres_min)):
            rms_out = 0
        elif ((Ngood > 5) & ((lx>2) & (lx < Nres_max)) & ((ly>2) & (ly < Nres_max))):
            zz = zdata
            zzfit = sum(zz)/len(zz)
            dz = zz-zzfit
            rms_out = np.sqrt(np.sum(dz**2)/Ngood)
        else:
            rms_out = 1000
    else:
        rms_out = 0
    return rms_out, Ngood, r_good

@cuda.jit
def quad_decomp_mean_parallel():
    pos = cuda.grid(1)
    cuda.syncthreads()