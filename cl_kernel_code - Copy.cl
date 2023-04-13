__kernel void multiply_arrays(__global float *a, __global float *b,
                              __global float *c) {
  size_t i = get_global_id(0);
  c[i] = a[i] * b[i];
  for (size_t j = 0; j < 4096; ++j)
    c[i] *= (c[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
}