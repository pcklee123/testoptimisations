Method 1:
    for (size_t i = 0; i < n; i++)
    {
        c1[i] = a[i] * b[i];
        for (size_t j = 0; j < 1024; j++)
            c1[i] *= (c1[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
    }
Method 2:
#pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        c2[i] = a[i] * b[i];
#pragma omp barrier
    for (size_t j = 0; j < 1024; j++)
#pragma omp parallel for
        for (size_t i = 0; i < n; i++)
            c2[i] *= (c2[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
#pragma omp barrier

Method 3:
    "__kernel void multiply_arrays(__global float* a, __global float* b, __global float* c) {\n"
    "   size_t i = get_global_id(0);\n"
    "   c[i] = a[i] * b[i];\n"
    "   for(size_t j=0;j<1024;++j) c[i] *= (c[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);"
    "}\n";