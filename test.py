import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule
import os
import time
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import pycuda.gpuarray as gpuarray
from plotting_utilities import *
import math

mod_quadtree =  SourceModule("""
    #include <cmath>
    #include "math.h"
    
    __device__ int find_ngood_bounds(int*zmask,int minx, int miny, int maxx, int maxy, int nx) {
        int ngood = 0;
        for (int i=miny; i<maxy; i++) {
            for (int j=minx; j<maxx; j++) {
                if (zmask[i+j*nx] == 1){
                    ngood ++;
                }
            }
        }
        return ngood;
    }
    
    __device__ double calc_avg_bounds(float *z, int*zmask,int minx, int miny, int maxx, int maxy, int nx){
        double sum_z = 0.0;
        int ngood = 0;
        for (int i=miny;i<maxy;i++) {
            for (int j=minx;j<maxx;j++) {
                if (zmask[i+j*nx] == 1) {
                    ngood += 1;
                    sum_z += z[i + j*nx];
                }
            }
        }
        return sum_z/ngood;
    }
    
    __device__ double calc_rms_bounds(float *z, int*zmask, int minx, int miny, int maxx, int maxy, int nx) {
        double z_avg = calc_avg_bounds(z, zmask, minx, miny, maxx, maxy, nx);
        int ngood = find_ngood_bounds(zmask, minx, miny, maxx, maxy, nx);
        double z_rms_sum = 0.0;
        for (int i=miny;i<maxy;i++) {
            for (int j=minx;j<maxx;j++) {
                if (zmask[i+j*nx]==1) {
                    z_rms_sum += pow(z[i+j*nx] - z_avg, 2);
                }
            }
        }
        return sqrt(z_rms_sum / ngood);
    }
    
    __device__ int find_unique_num(float *arr, int mina, int maxa) {
        int uniq_s1 = maxa - mina;
        for (int i=0; i < maxa - mina; i++)
        {
            for (int j=0; j<i; j++){
                if (arr[i] == arr[j]){
                    uniq_s1 -= 1;
                    break;
                }      
            }
        }
        return uniq_s1;
    }
    
    __device__ double rms_block_demean_bounds(float *x, float *y, float*z, int*zmask, int Nres_min,
                            int Nres_max, int minx, int miny, int maxx, int maxy, int nx){
        int Ngood = find_ngood_bounds(zmask, minx, miny, maxx, maxy, nx);
        int n_block = (maxx-minx) * (maxy-miny);
        double r_good = Ngood*1.0/n_block;
        double rms_out = 0;
        if (Ngood > 0){
            int lx = find_unique_num(x, minx, maxx);
            int ly = find_unique_num(y, miny, maxy);
            if ((Ngood<=3) | (lx<=Nres_min) | (ly<=Nres_min)){
                rms_out = 0;
            }else{
                if ((Ngood > 5) & ((lx>2) & (lx < Nres_max)) & ((ly>2) & (ly < Nres_max))) {
                    rms_out = calc_rms_bounds(z, zmask, minx, miny, maxx, maxy, nx);
                }
                else {
                    rms_out = 1000;
                }
            }
        }
        return rms_out;
    }
    
    __global__ void construct_new_nodes(float*xin, float*yin, float *zin, int* zinmask, int*minx, int*miny, 
    int*maxx, int*maxy, int*isnode, int*isleaf, int ny, int nx, 
    float threshold, int Nres_min, int Nres_max, int max_depth, int psindx, int peindx, 
    int* csindx, int* this_layer_node, int interval, float* rgoodin) {
        double r_good_default = 0.2;
        
        int idx = threadIdx.x + threadIdx.y*6 ;
        int minrange = psindx + idx*interval;

        if (minrange < peindx){
            int maxrange = min(psindx + (idx+1)*interval, peindx);
            for (int i_node = minrange; i_node < maxrange; i_node ++){
                if (isnode[i_node] == 1) {
                    if (isleaf[i_node] == 0){
                        int nx1 = minx[i_node];
                        int xincrement = int((maxx[i_node] - minx[i_node])/2);
                        int nx2 = nx1 + xincrement;
                        int nx3 = nx2;
                        int nx4 = maxx[i_node];
        
                        int ny1 = miny[i_node];
                        int yincrement = int((maxy[i_node] - miny[i_node])/2);
                        int ny2 = ny1 + yincrement;
                        int ny3 = ny2;
                        int ny4 = maxy[i_node];
                        
                        int thiscsindx = atomicAdd(&csindx[0], 4);
                        minx[thiscsindx+1] = nx1, minx[thiscsindx+2] = nx3, minx[thiscsindx+3] = nx1, minx[thiscsindx+4] = nx3;
                        maxx[thiscsindx+1] = nx2, maxx[thiscsindx+2] = nx4, maxx[thiscsindx+3] = nx2, maxx[thiscsindx+4] = nx4;
                        miny[thiscsindx+1] = ny1, miny[thiscsindx+2] = ny1, miny[thiscsindx+3] = ny3, miny[thiscsindx+4] = ny3;
                        maxy[thiscsindx+1] = ny2, maxy[thiscsindx+2] = ny2, maxy[thiscsindx+3] = ny4, maxy[thiscsindx+4] = ny4;
                        
                        atomicAdd(this_layer_node, 4);
                        for (int j_index = 1; j_index < 5; j_index++) {
                            int this_index = thiscsindx+j_index;
                            int this_nx = maxx[this_index] - minx[this_index];
                            int this_ny = maxy[this_index] - miny[this_index];
                            double rms =  rms_block_demean_bounds(xin, yin, zin, zinmask, Nres_min,
                                    Nres_max,  minx[this_index],  miny[this_index],
                                    maxx[this_index],  maxy[this_index], nx);
                            int ngood = find_ngood_bounds(zinmask, minx[this_index],  miny[this_index],  
                                    maxx[this_index],  maxy[this_index], nx);
                            float r_good = ngood*1.0/(this_nx*this_ny*1.0);
                            bool leaf_crit1 = ((rms <= threshold) & (ngood > 0) & (r_good > r_good_default));
                            bool leaf_crit2 = ((this_nx <= Nres_min) | (this_ny<=Nres_min));
                            this_layer_node[1] = ngood;
                            rgoodin[0] = rms;
                            if (leaf_crit1 | leaf_crit2){
                                if ((ngood > 0) & (r_good > r_good_default)){
                                    isleaf[this_index] = 1;
                                    isnode[this_index] = 1;
                                    
                                }else {
                                    isnode[this_index] = 0;
                                    isleaf[this_index] = 0;
                                }
                            }else{
                                isnode[this_index] = 1;
                                isleaf[this_index] = 0;
                            }
                        }
                    }
                }
            }
        }
    }
    
      
      """)

def test_cuda():
    cuda.init()
    assert cuda.Device.count() >= 1
    print(cuda.Device.count())
    #dev = cuda.Device(0)
    #ctx = cuda.make_context()

    mod = SourceModule("""
      __global__ void doublify(float *a)
      {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
      }
      """)
    a = np.random.randn(4,4).astype(np.float32)
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu, a)
    func = mod.get_function("doublify")
    func(a_gpu, block=(4, 4, 1))
    a_doubled = np.empty_like(a)
    cuda.memcpy_dtoh(a_doubled, a_gpu)


def main():
    # Read-in the data: Ridgecrest earthquake
    # Original dataset: matrix size 2330*3960 "Ridgecrest_ori.grd"
    # Smaller version: matrix size 1320*777 "Ridgecrest_small.grd"
    grid=nc.Dataset("Ridgecrest_small.grd")
    xinsar=np.array(grid['lon']).astype(np.float32) # 777
    yinsar=np.array(grid['lat']).astype(np.float32) # 1320
    zinsar=np.array(grid['z']).astype(np.float32) # 1320*777
    zinmask = np.int32(~np.isnan(zinsar))
    print(np.sum(zinmask[:]))
    Nmin = 1000# 1000 # minimum number of downsampled grids
    Nres_min = 10#10 # minimum grid size
    Nres_max = 1000 # maxium grid size
    xlen = len(xinsar)
    ylen = len(yinsar)

    xin_gpu = gpuarray.to_gpu(xinsar)
    yin_gpu = gpuarray.to_gpu(yinsar)
    zin_gpu = gpuarray.to_gpu(zinsar)
    zinmask_gpu = gpuarray.to_gpu(zinmask)
    z_avg = gpuarray.to_gpu(np.zeros(36).astype(np.int32))

    rms_default = 10
    r_good_default = 0.2
    max_depth = int(math.log((xlen*ylen)/(Nres_min**2), 4))+2
    node_total = int(((xlen*ylen)/(Nres_min*Nres_min))+1)
    minx = gpuarray.to_gpu(np.zeros(node_total).astype(np.int32))
    miny = gpuarray.to_gpu(np.zeros(node_total).astype(np.int32))
    maxx_cpu = np.zeros(node_total)
    maxx_cpu[0] = xlen
    maxx = gpuarray.to_gpu(maxx_cpu.astype(np.int32))
    maxy_cpu = np.zeros(node_total)
    maxy_cpu[0] = ylen
    maxy = gpuarray.to_gpu(maxy_cpu.astype(np.int32))
    isnode_cpu = np.zeros(node_total)
    isnode_cpu[0] = 1
    isnode = gpuarray.to_gpu(isnode_cpu.astype(np.int32))
    isleaf = gpuarray.to_gpu(np.zeros(node_total).astype(np.int32))
    this_layer_node = gpuarray.to_gpu(np.zeros(3).astype(np.int32))
    csindx_input = gpuarray.to_gpu(np.zeros(1).astype(np.int32))
    rgood = gpuarray.to_gpu(np.zeros(1).astype(np.float32))

    psindx = 0
    peindx = np.int32(1)

    func = mod_quadtree.get_function("construct_new_nodes")
    for i_depth in range(max_depth - 1):
        prev = this_layer_node.get()[0]
        interval = int((peindx - psindx) / 36) + 1
        print(interval)
        func(xin_gpu, yin_gpu, zin_gpu, zinmask_gpu, minx, miny, maxx, maxy, isnode, isleaf,
             np.int32(ylen), np.int32(xlen),
             np.int32(10), np.int32(Nres_min), np.int32(Nres_max), np.int32(max_depth),
             np.int32(psindx), np.int32(peindx), csindx_input, this_layer_node, np.int32(interval),rgood,
             block=(6, 6, 1))
        #print('this_layer_node')
        #print(this_layer_node.get())
        #print('rgood : {}'.format(rgood.get()[0]))

        #print(isnode.get()[:this_layer_node.get()[0]])
        #print(isleaf.get()[:this_layer_node.get()[0]])
        psindx = peindx
        peindx = peindx + this_layer_node.get()[0] - prev

    minx_cpu = minx.get()
    maxx_cpu = maxx.get()
    miny_cpu = miny.get()
    maxy_cpu = maxy.get()
    isnode_cpu = isnode.get()
    isleaf_cpu = isleaf.get()

    xout = []
    yout = []
    zout = []
    rms_out = []
    Ndata = []
    xx1 = []
    xx2 = []
    yy1 = []
    yy2 = []
    for i_node in range(node_total):
        if (isleaf_cpu[i_node] & isnode_cpu[i_node]):
            #print('this depth is {}, this node is {}'.format(i_depth, i_node))
            this_z = zinsar[miny_cpu[i_node]:maxy_cpu[i_node],
                     minx_cpu[i_node]:maxx_cpu[i_node]]
            this_x = xinsar[minx_cpu[i_node]:maxx_cpu[i_node]]
            this_y = yinsar[miny_cpu[i_node]:maxy_cpu[i_node]]
            z_block_good = this_z[~np.isnan(this_z)]
            N_block_good = len(z_block_good)
            if N_block_good > 0:
                xout_block_total = sum(this_x)/len(this_x)
                yout_block_total = sum(this_y)/len(this_y)
                n_block = len(this_x) * len(this_y)
                r_good = N_block_good / n_block
                zout_block_total = sum(z_block_good)/len(z_block_good)
                dz_block = z_block_good - zout_block_total
                rms_block_total = np.sqrt((sum(dz_block ** 2) / N_block_good))
                #if (rms_block_total < 1.0e-6):
                #    rms_block_total = 10
                xout.append(xout_block_total)
                yout.append(yout_block_total)
                zout.append(zout_block_total)
                rms_out.append(rms_block_total)
                Ndata.append(N_block_good)
                xx1.append(xinsar[minx_cpu[i_node]])
                xx2.append(xinsar[maxx_cpu[i_node]-1])
                yy1.append(yinsar[miny_cpu[i_node]])
                yy2.append(yinsar[maxy_cpu[i_node]-1])
    print(rms_out)
    cmin = np.min(zout)
    cmax = np.max(zout)

    xout = np.array(xout)
    yout = np.array(yout)
    zout = np.array(zout)

    xx1 = np.array(xx1)
    xx2 = np.array(xx2)
    yy1 = np.array(yy1)
    yy2 = np.array(yy2)
    # plot_insar_data_scatter(xout,yout,zout,image_name);
    #plot_insar_sample(xout, yout, zinsar, zout, xx1, xx2, yy1, yy2, cmin, cmax, 'test_recursive_small.png');



if __name__=='__main__':
    main()


