#define nn 4096
__kernel void method1(__global float4 *a, __global float4 *b,
                      __global float4 *c) {
  size_t i = get_global_id(0);
  float4 va = a[i];
  float4 vb = b[i];
  float4 vc = va * vb;

  float4 one_vec = (float4)(1.0f);
  for (size_t j = 0; j < 4096; j += 4) {
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

__kernel void method2(__global const float4 *a, __global const float4 *b,
                      __global float4 *c2) {
  size_t i = get_global_id(0);
  float4 result = a[i] * b[i];
  float4 ab1 = (a[i] + 1.0f) * (b[i] + 1.0f);
  for (size_t j = 0; j < 4096; j++) {
    result *= (result + 1.0f) * ab1;
  }
  c2[i] = result;
}

__kernel void method3(__global const float4 *a, __global const float4 *b,
                      __global float4 *c2) {
  size_t i = get_global_id(0);
  float4 ab1 = (a[i] + 1.0f) * (b[i] + 1.0f);
  for (size_t j = 0; j < 4096; j++) {
    float4 temp = a[i] * b[i];
    temp *= (temp + 1.0f) * ab1;
    c2[i] += temp;
  }
}