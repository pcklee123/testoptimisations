__kernel void multiply_arrays(__global float4* a, __global float4* b, __global float4* c) {
    size_t i = get_global_id(0);
    float4 a_vec = a[i];
    float4 b_vec = b[i];
    float4 c_vec = a_vec * b_vec;
    
    float4 one_vec = (float4)(1.0f);
    for (size_t j = 0; j < 4096; j += 4) {
        float4 temp_vec = c_vec + one_vec;
        temp_vec *= (a_vec + one_vec) * (b_vec + one_vec);
        c_vec *= temp_vec;

        temp_vec = c_vec + one_vec;
        temp_vec *= (a_vec + one_vec) * (b_vec + one_vec);
        c_vec *= temp_vec;

        temp_vec = c_vec + one_vec;
        temp_vec *= (a_vec + one_vec) * (b_vec + one_vec);
        c_vec *= temp_vec;

        temp_vec = c_vec + one_vec;
        temp_vec *= (a_vec + one_vec) * (b_vec + one_vec);
        c_vec *= temp_vec;
    }

    c[i] = c_vec;
}