# Test various methods

## Method 1:

    for (size_t i = 0; i < n; i++)
    {
        c1[i] = a[i] * b[i];
        for (size_t j = 0; j < 4096; j++)
            c1[i] *= (c1[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
    }

## Method 2:

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        c2[i] = a[i] * b[i];
    #pragma omp barrier
    for (size_t j = 0; j < 4096; j++)
    #pragma omp parallel for
        for (size_t i = 0; i < n; i++)
            c2[i] *= (c2[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
    #pragma omp barrier

## Method 3:

    __kernel void multiply_arrays(__global float *a, __global float *b,__global float *c) {
      size_t i = get_global_id(0);
      c[i] = a[i] * b[i];
      for (size_t j = 0; j < 4096; ++j)
        c[i] *= (c[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0); 
    }


## results for  n=1024*1024;
Method 1 Elapsed time: 16642 ms  
Method 2 Elapsed time: 390 ms  

source loaded  
Method 3 Elapsed time: 322 ms thread0: Using OpenCL device: gfx90c  
Method 3 Elapsed time: 173 ms thread1: Using OpenCL device: gfx1032
Method 3 Elapsed time: 172 ms thread2: Using OpenCL device: gfx1032