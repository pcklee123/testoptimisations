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
#pragma omp parallel for simd
    for (size_t i = 0; i < n; i++)
    {
        c1[i] = a[i] * b[i];
#pragma omp simd
        for (size_t j = 0; j < nn; j++)
            c1[i] *= (c1[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
    }
### 1c With AVX2
    for (size_t i = 0; i < n; i += VEC_WIDTH)
    {
        va = _mm256_loadu_ps(&a[i]);
        vb = _mm256_loadu_ps(&b[i]);
        vc = _mm256_mul_ps(va, vb);
        for (size_t j = 1; j < nn; j++)
        {
            v1 = _mm256_add_ps(vc, one_vec);
            v1 = _mm256_mul_ps(v1, _mm256_add_ps(va, one_vec));
            v1 = _mm256_mul_ps(v1, _mm256_add_ps(vb, one_vec));
            vc = _mm256_mul_ps(vc, v1);
        }
        _mm256_storeu_ps(&c2[i], vc);
    }
### 1d With OpenCL
    #define nn 4096
    __kernel void method1(__global const float4 *a, __global const float4 *b,
                        __global float4 *c) {
    size_t i = get_global_id(0);
    float4 va = a[i];
    float4 vb = b[i];
    float4 vc = va * vb;

    float4 one_vec = (float4)(1.0f);
    for (size_t j = 0; j < nn; j += 4) {
        float4 v1 = vc + one_vec;
        v1 *= (va + one_vec) * (vb + one_vec);
        vc *= v1;

        v1 = vc + one_vec;
        v1 *= (va + one_vec) * (vb + one_vec);
        vc *= v1;

        v1 = vc + one_vec;
        v1 *= (va + one_vec) * (vb + one_vec);
        vc *= v1;

        v1 = vc + one_vec;
        v1 *= (va + one_vec) * (vb + one_vec);
        vc *= v1;
    }

    c[i] = vc;
    }
## Method 2:

### 2a Base code   
    for (size_t i = 0; i < n; i++)
        c2[i] = a[i] * b[i];
    for (size_t j = 0; j < nn; j++)
        for (size_t i = 0; i < n; i++)
            c2[i] *= (c2[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
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
    for (size_t j = 0; j < nn; j++)
    {
        for (size_t i = 0; i < n; i++)
        {
            temp = a[i] * b[i];
            temp *= (temp + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
            c2[i] += temp;
        }
    }
### 3b With OpenMP directives  

### 3c With AVX2

### 3d With OpenCL
## results for  n=1024*1024;
