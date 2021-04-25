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

mod_quadtree =  SourceModule("""

    __device__ double calc_avg(float **z, int**zmask, int size1, int size2){
        double sum_z = 0.0;
        int ngood = 0;
        for (int i=0;i<size1;i++) {
            for (int j=0;j<size2;j++) {
                if (zmask[i][j] == 1) {
                    ngood ++;
                    sum_z += z[i][j];
                }
            }
        }
        return sum_z/ngood;
    }
    
    __global__ void try_gpu(float **z, int**zmask, int size1, int size2, float*z_avg) {
        int idx = threadIdx.x + threadIdx.y*4;
        float sum_z = 0.0;
        int ngood = 0;
        for (int i=0;i<size1;i++) {
            for (int j=0;j<size2;j++) {
                if (zmask[i][j] == 1) {
                    ngood ++;
                    sum_z += z[i][j];
                }
            }
        }
        z_avg[idx] =sum_z/ngood;
    }
    
    
      __global__ void quad_decomp_const(float *xin, float *yin, float *zin)
      {
        int idx = threadIdx.x + threadIdx.y*4;
        zin[idx] *= 2;
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
    zinmask = ~np.isnan(zinsar).astype(np.int)

    Nmin = 1000# 1000 # minimum number of downsampled grids
    Nres_min = 10#10 # minimum grid size
    Nres_max = 400 # maxium grid size
    xlen = len(xinsar)
    ylen = len(yinsar)

    xin_gpu = gpuarray.to_gpu(xinsar)
    yin_gpu = gpuarray.to_gpu(yinsar)
    zin_gpu = gpuarray.to_gpu(zinsar)
    zinmask_gpu = gpuarray.to_gpu(zinmask)
    z_avg = gpuarray.to_gpu(np.zeros(16).astype(np.float32))

    xlen_gpu = cuda.mem_alloc(np.int.nbytes)
    ylen_gpu = cuda.mem_alloc(np.int.nbytes)
    cuda.memcpy_dtoh(xlen, xlen_gpu)
    cuda.memcpy_dtoh(xlen, ylen_gpu)
    #xlen_gpu = gpuarray.to_gpu(xlen)
    #ylen_gpu = gpuarray.to_gpu(ylen)

    print(xlen)
    print(ylen)
    print(z_avg)
    #z_avg = 0.0
    func = mod_quadtree.get_function("try_gpu")
    func(zin_gpu, zinmask_gpu, ylen_gpu, xlen_gpu, z_avg, block=(1, 1, 1))
    print(z_avg.get())





if __name__=='__main__':
    main()


