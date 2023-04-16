#include "main.h"
#define nn 4096
int main()
{
    const int VEC_WIDTH = 8; // AVX256 supports 8 float elements per vector
    const size_t n = 1024 * 128;
    // allocate an array of n integers aligned to a 32-byte(256bit) boundary
    int boun = 32;
    float *a = (float *)_mm_malloc(n * sizeof(float), boun);
    float *b = (float *)_mm_malloc(n * sizeof(float), boun);
    float *c1 = (float *)_mm_malloc(n * sizeof(float), boun);
    float *c2 = (float *)_mm_malloc(n * sizeof(float), boun);
    float *c3_ptr = (float *)_mm_malloc(max_cl_dev * n * sizeof(float), 64);
    float(*c3)[n] = (float(*)[n])c3_ptr;
    __m256 va, vb, vc, v1, one_vec;
    //    __m256 va, vb, vc, v1, v2, v3, v4, one_vec;
    one_vec = _mm256_set1_ps(1.0f);
    float temp;

    // Initialize a and b with random values
    auto start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for simd
    for (size_t i = 0; i < n; i++)
    {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }
    // load cl program source from file
    std::ifstream t("cl_kernel_code.cl");
    std::string kernel_code((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
    const char *kernel_source = kernel_code.c_str();
    const char **source = &kernel_source;
    std::cout << "source loaded " << std::endl;
    //  Get the platform
    status1 = clGetPlatformIDs(1, &platform, NULL);
    // Get the device
    // num_devices=5;
    status1 = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, max_cl_dev, device, &num_devices);
    // std::cout << "status " << status << std::endl;
    // std::cout << "num_devices " << num_devices << std::endl;
    char device_name[max_cl_dev][20];
#pragma omp parallel for
    for (cl_uint j = 0; j < num_devices; j++)
    {
        cl_int status;
        //       cl_int err;
        // Get the device name
        size_t device_name_size;
        clGetDeviceInfo(device[j], CL_DEVICE_NAME, 0, NULL, &device_name_size);
        // std::cout << "device_name_size " << device_name_size << std::endl;
        //  char *device_name[j] = new char[j][device_name_size];
        clGetDeviceInfo(device[j], CL_DEVICE_NAME, device_name_size, device_name[j], NULL);
        //       std::cout << j << ": Using OpenCL device: " << device_name << std::endl;

        // Create the context
        context[j] = clCreateContext(NULL, 1, &device[j], NULL, NULL, &status);

        // Create the program

        program[j] = clCreateProgramWithSource(context[j], 1, source, NULL, &status);

        // Build the program
        status = clBuildProgram(program[j], 1, &device[j], NULL, NULL, NULL);

        // Create the kernel
        kernel1[j] = clCreateKernel(program[j], "method1", &status);
        // Create the kernel
        kernel2[j] = clCreateKernel(program[j], "method2", &status);
        // Create the kernel
        kernel3[j] = clCreateKernel(program[j], "method3", &status);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Start up Elapsed time: " << elapsed_time.count() << " ms" << std::endl;

    // method 1
    //  Perform the element-wise calculations do all calculations for 1 element before moving on to the next element
    start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < n; i++)
    {
        c1[i] = a[i] * b[i];
        for (size_t j = 0; j < nn; j++)
            c1[i] *= (c1[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
    }

    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Method 1a Elapsed time: " << elapsed_time.count() << " ms" << std::endl;

    // method 1b
    //  Perform the element-wise calculations do all calculations for 1 element before moving on to the next element
    start_time = std::chrono::high_resolution_clock::now();

#pragma omp parallel for simd
    for (size_t i = 0; i < n; i++)
    {
        c1[i] = a[i] * b[i];
#pragma omp simd
        for (size_t j = 0; j < nn; j++)
            c1[i] *= (c1[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
    }

    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Method 1b with openmp Elapsed time: " << elapsed_time.count() << " ms" << std::endl;

    // Method 1c
    // do all elements and store the result in c2 before going on to the next step of calculations
    start_time = std::chrono::high_resolution_clock::now();

#pragma omp parallel for private(va, vb, vc, v1)
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
#pragma omp barrier
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Method 1c AVX2 version of Method 1 Elapsed time: " << elapsed_time.count() << " ms" << std::endl;

    // print differences
    for (size_t i = 0; i < n; i++)
    {
        if (c2[i] != c1[i])
            std::cout << i << ",";
    }

    // method 1d
    //  #pragma omp parallel for
    for (cl_uint j = 0; j < num_devices; j++)
    {
        cl_int status;
        cl_int err;
        // In real situations the above is run once and the kernel is used without building from the source later.
        start_time = std::chrono::high_resolution_clock::now();
        // Create the command queue
        cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        queue[j] = clCreateCommandQueueWithProperties(context[j], device[j], properties, &err);

        // Create the buffers
        bufferA[j] = clCreateBuffer(context[j], CL_MEM_READ_ONLY, n * sizeof(float), NULL, &status);
        bufferB[j] = clCreateBuffer(context[j], CL_MEM_READ_ONLY, n * sizeof(float), NULL, &status);
        bufferC[j] = clCreateBuffer(context[j], CL_MEM_READ_WRITE, n * sizeof(float), NULL, &status);

        // Copy data to the buffers
        status = clEnqueueWriteBuffer(queue[j], bufferA[j], CL_TRUE, 0, n * sizeof(float), a, 0, NULL, NULL);
        status = clEnqueueWriteBuffer(queue[j], bufferB[j], CL_TRUE, 0, n * sizeof(float), b, 0, NULL, NULL);
        // Set the kernel arguments
        status = clSetKernelArg(kernel1[j], 0, sizeof(cl_mem), (void *)&bufferA[j]);
        status = clSetKernelArg(kernel1[j], 1, sizeof(cl_mem), (void *)&bufferB[j]);
        status = clSetKernelArg(kernel1[j], 2, sizeof(cl_mem), (void *)&bufferC[j]);

        // Execute the kernel
        size_t globalWorkSize[1] = {n};
        status = clEnqueueNDRangeKernel(queue[j], kernel1[j], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        // Read the result from the buffer
        status = clEnqueueReadBuffer(queue[j], bufferC[j], CL_TRUE, 0, n * sizeof(float), c3[j], 0, NULL, NULL);
        end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Method 1d opencl Elapsed time: " << elapsed_time.count() << " ms thread" << j << ": Using OpenCL device: " << device_name[j] << std::endl;

        // print differences
        for (size_t i = 0; i < n; i++)
        {
            if (c1[i] != c3[j][i])
                std::cout << i << ",";
        }
        std::cout << std::endl;
    }

    // Method 2a
    //  do all elements and store the result in c before going on to the next step of calculations
    start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < n; i++)
        c2[i] = a[i] * b[i];
    for (size_t j = 0; j < nn; j++)
        for (size_t i = 0; i < n; i++)
            c2[i] *= (c2[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);

    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Method 2a Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
    // print differences
    for (size_t i = 0; i < n; i++)
    {
        if (c2[i] != c1[i])
            std::cout << i << ",";
    }

    // Method 2b
    //  do all elements and store the result in c before going on to the next step of calculations
    start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for simd
    for (size_t i = 0; i < n; i++)
        c2[i] = a[i] * b[i];
#pragma omp barrier
#pragma omp parallel for
    for (size_t j = 0; j < nn; j++)
#pragma omp simd
        for (size_t i = 0; i < n; i++)
            c2[i] *= (c2[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
#pragma omp barrier
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Method 2b with OpenMP Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
    // print differences
    for (size_t i = 0; i < n; i++)
    {
        if (c2[i] != c1[i])
            std::cout << i << ",";
    }

    // Method 2c
    //  do all elements and store the result in c before going on to the next step of calculations
    start_time = std::chrono::high_resolution_clock::now();
    //   __m256 va, vb, vc, v1, v2, v3, v4;
#pragma omp parallel for private(va, vb, vc)
    // #pragma omp parallel for
    for (size_t i = 0; i < n; i += VEC_WIDTH)
    {
        va = _mm256_load_ps(&a[i]);
        vb = _mm256_load_ps(&b[i]);
        vc = _mm256_mul_ps(va, vb);
        _mm256_store_ps(&c2[i], vc);
    }
#pragma omp barrier

    for (size_t j = 0; j < nn; j++)
    {
#pragma omp parallel for private(va, vb, vc, v1)
        for (size_t i = 0; i < n; i += VEC_WIDTH)
        {
            va = _mm256_load_ps(&a[i]);
            vb = _mm256_load_ps(&b[i]);
            vc = _mm256_load_ps(&c2[i]);
            v1 = _mm256_fmadd_ps(vc, _mm256_add_ps(va, one_vec), one_vec);
            v1 = _mm256_fmadd_ps(v1, _mm256_add_ps(vb, one_vec), one_vec);
            vc = _mm256_mul_ps(vc, v1);
            _mm256_store_ps(&c2[i], vc);
        }
#pragma omp barrier
    }

    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Method 2c with avx 2 Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
    // print differences
    for (size_t i = 0; i < n; i++)
    {
        if (c2[i] != c1[i])
            std::cout << i << ",";
    }

    // method 2d
    //  #pragma omp parallel for
    for (cl_uint j = 0; j < num_devices; j++)
    {
        cl_int status;
        cl_int err;
        // In real situations the above is run once and the kernel is used without building from the source later.
        start_time = std::chrono::high_resolution_clock::now();
        // Create the command queue
        cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        queue[j] = clCreateCommandQueueWithProperties(context[j], device[j], properties, &err);

        // Create the buffers
        bufferA[j] = clCreateBuffer(context[j], CL_MEM_READ_ONLY, n * sizeof(float), NULL, &status);
        bufferB[j] = clCreateBuffer(context[j], CL_MEM_READ_ONLY, n * sizeof(float), NULL, &status);
        bufferC[j] = clCreateBuffer(context[j], CL_MEM_READ_WRITE, n * sizeof(float), NULL, &status);

        // Copy data to the buffers
        status = clEnqueueWriteBuffer(queue[j], bufferA[j], CL_TRUE, 0, n * sizeof(float), a, 0, NULL, NULL);
        status = clEnqueueWriteBuffer(queue[j], bufferB[j], CL_TRUE, 0, n * sizeof(float), b, 0, NULL, NULL);
        // Set the kernel arguments
        status = clSetKernelArg(kernel2[j], 0, sizeof(cl_mem), (void *)&bufferA[j]);
        status = clSetKernelArg(kernel2[j], 1, sizeof(cl_mem), (void *)&bufferB[j]);
        status = clSetKernelArg(kernel2[j], 2, sizeof(cl_mem), (void *)&bufferC[j]);

        // Execute the kernel
        size_t globalWorkSize[1] = {n};
        status = clEnqueueNDRangeKernel(queue[j], kernel2[j], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        // Read the result from the buffer
        status = clEnqueueReadBuffer(queue[j], bufferC[j], CL_TRUE, 0, n * sizeof(float), c3[j], 0, NULL, NULL);
        end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Method 2d opencl Elapsed time: " << elapsed_time.count() << " ms thread" << j << ": Using OpenCL device: " << device_name[j] << std::endl;

        // print differences
        for (size_t i = 0; i < n; i++)
        {
            if (c1[i] != c3[j][i])
                std::cout << i << ",";
        }
        std::cout << std::endl;
    }

    // Method 3a
    // do calculations element by element and store the result in c1
    start_time = std::chrono::high_resolution_clock::now();

    for (size_t j = 0; j < nn; j++)
    {
        for (size_t i = 0; i < n; i++)
        {
            temp = a[i] * b[i];
            temp *= (temp + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
            c2[i] += temp;
        }
    }

    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Method 3a Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
    // print differences
    for (size_t i = 0; i < n; i++)
    {
        if (c2[i] != c1[i])
            std::cout << i << ",";
    }

    // Method 3b
    // do calculations element by element and store the result in c1
    start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for private(temp)
    for (size_t j = 0; j < nn; j++)
    {
        for (size_t i = 0; i < n; i++)
        {
            temp = a[i] * b[i];
            temp *= (temp + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
            c2[i] += temp;
        }
    }
#pragma omp barrier
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Method 3b OpenMP Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
    // print differences
    for (size_t i = 0; i < n; i++)
    {
        if (c2[i] != c1[i])
            std::cout << i << ",";
    }
    // Method 3c
    // do calculations element by element and store the result in c1
    start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for private(va, vb, vc, v1)
    for (size_t j = 0; j < nn; j++)
    {
        for (size_t i = 0; i < n; i += 8)
        {
            va = _mm256_loadu_ps(&a[i]);
            vb = _mm256_loadu_ps(&b[i]);
            v1 = _mm256_mul_ps(va, vb);
            v1 = _mm256_fmadd_ps(v1, v1, one_vec);
            v1 = _mm256_fmadd_ps(v1, va, one_vec);
            v1 = _mm256_fmadd_ps(v1, vb, one_vec);
            vc = _mm256_loadu_ps(&c2[i]);
            vc = _mm256_add_ps(vc, v1);
            _mm256_storeu_ps(&c2[i], vc);
        }
    }
#pragma omp barrier
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Method 3c AVX2 Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
    // print differences
    for (size_t i = 0; i < n; i++)
    {
        if (c2[i] != c1[i])
            std::cout << i << ",";
    }
    std::cout << std::endl;
/*
    // method 3d
    //  #pragma omp parallel for
    for (cl_uint j = 0; j < num_devices; j++)
    {
        cl_int status;
        cl_int err;
        // In real situations the above is run once and the kernel is used without building from the source later.
        start_time = std::chrono::high_resolution_clock::now();
        // Create the command queue
        cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        queue[j] = clCreateCommandQueueWithProperties(context[j], device[j], properties, &err);

        // Create the buffers
        bufferA[j] = clCreateBuffer(context[j], CL_MEM_READ_ONLY, n * sizeof(float), NULL, &status);
        bufferB[j] = clCreateBuffer(context[j], CL_MEM_READ_ONLY, n * sizeof(float), NULL, &status);
        bufferC[j] = clCreateBuffer(context[j], CL_MEM_WRITE_ONLY, n * sizeof(float), NULL, &status);

        // Copy data to the buffers
        status = clEnqueueWriteBuffer(queue[j], bufferA[j], CL_TRUE, 0, n * sizeof(float), a, 0, NULL, NULL);
        status = clEnqueueWriteBuffer(queue[j], bufferB[j], CL_TRUE, 0, n * sizeof(float), b, 0, NULL, NULL);
        // Set the kernel arguments
        status = clSetKernelArg(kernel3[j], 0, sizeof(cl_mem), (void *)&bufferA[j]);
        status = clSetKernelArg(kernel3[j], 1, sizeof(cl_mem), (void *)&bufferB[j]);
        status = clSetKernelArg(kernel3[j], 2, sizeof(cl_mem), (void *)&bufferC[j]);

        // Execute the kernel
        size_t globalWorkSize[1] = {n};
        status = clEnqueueNDRangeKernel(queue[j], kernel3[j], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        // Read the result from the buffer
        status = clEnqueueReadBuffer(queue[j], bufferC[j], CL_TRUE, 0, n * sizeof(float), c3[j], 0, NULL, NULL);
        end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Method 3d opencl Elapsed time: " << elapsed_time.count() << " ms thread" << j << ": Using OpenCL device: " << device_name[j] << std::endl;

        // print differences
        for (size_t i = 0; i < n; i++)
        {
            if (c1[i] != c3[j][i])
            {
                std::cout << i << ",";
                break;
            }
        }
        std::cout << std::endl;
    }
    */
    return 0;
}
