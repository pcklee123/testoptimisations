__kernel void multiply_arrays(__global float4* a, __global float4* b, __global float4* c) {
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