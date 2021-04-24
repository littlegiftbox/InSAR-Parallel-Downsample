import pycuda.driver as cuda
import numpy as np


mod = cuda.SourceModule("""
      __global__ void doublify(float *a)
      {
        int idx = threadIdx.x + threadIdx.y*4;
        a[idx] *= 2;
      }
      """)

def test_cuda():
    cuda.init()
    assert cuda.Device.count() >= 1
    dev = cuda.Device(0)
    ctx = dev.make_context()
    a = np.random.randn(4,4).astype(np.float32)
    a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)
    cuda.memcpy_htod(a_gpu, a)
    func = mod.get_function("doublify")
    func(a_gpu, block=(4, 4, 1))
    a_doubled = np.empty_like(a)
    cuda.memcpy_dtoh(a_doubled, a_gpu)







