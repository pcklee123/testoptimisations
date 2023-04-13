# Test coding styles

## Method 1:
### 1a Base code 
    for (size_t i = 0; i < n; i++)
    {
        c1[i] = a[i] * b[i];
        for (size_t j = 0; j < 4096; j++)
            c1[i] *= (c1[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
    }

### 1b With OpenMP directives

### 1c With AVX2

### 1d With OpenCL
## Method 2:

### 2a Base code   
### 2b With OpenMP directives  

    #pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        c2[i] = a[i] * b[i];
    #pragma omp barrier
    for (size_t j = 0; j < 4096; j++)
    #pragma omp parallel for
        for (size_t i = 0; i < n; i++)
            c2[i] *= (c2[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
    #pragma omp barrier
### 2c With AVX2

### 2d With OpenCL
    __kernel void multiply_arrays(__global float *a, __global float *b,__global float *c) {
      size_t i = get_global_id(0);
      c[i] = a[i] * b[i];
      for (size_t j = 0; j < 4096; ++j)
        c[i] *= (c[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0); 
    }
## Method 3:

### 3a Base code   
### 3b With OpenMP directives  

### 3c With AVX2

### 3d With OpenCL
## results for  n=1024*1024;
