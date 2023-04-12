#include "main.h"

const char *kernelSource =
    "__kernel void multiply_arrays(__global float* a, __global float* b, __global float* c) {\n"
    "   size_t i = get_global_id(0);\n"
    "   c[i] = a[i] * b[i];\n"
    "   for(size_t j=0;j<1024;++j) c[i] *= (c[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);"
    "}\n";

int main()
{
    const size_t n = 1024 * 1024;
    // allocate an array of n integers aligned to a 32-byte(256bit) boundary
    float *a = (float *)_mm_malloc(n * sizeof(float), 32);
    float *b = (float *)_mm_malloc(n * sizeof(float), 32);
    float *c1 = (float *)_mm_malloc(n * sizeof(float), 32);
    float *c2 = (float *)_mm_malloc(n * sizeof(float), 32);
    float *c3 = (float *)_mm_malloc(n * sizeof(float), 32);
// Initialize a and b with random values
#pragma omp parallel for simd
    for (size_t i = 0; i < n; i++)
    {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }

    // Perform the element-wise calculations do all calculations for 1 element before moving on to the next element
    auto start_time = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < n; i++)
    {
        c1[i] = a[i] * b[i];
        for (size_t j = 0; j < 1024; j++)
            c1[i] *= (c1[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Method 1 Elapsed time: " << elapsed_time.count() << " ms" << std::endl;

    // do all elements and store the result in c before going on to the next step of calculations
    start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        c2[i] = a[i] * b[i];
#pragma omp barrier
    for (size_t j = 0; j < 1024; j++)
#pragma omp parallel for
        for (size_t i = 0; i < n; i++)
            c2[i] *= (c2[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
#pragma omp barrier
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Method 2 Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
    // print differences
    for (size_t i = 0; i < n; i++)
    {
        if (c2[i] != c1[i])
            std::cout << i << ",";
    }
    std::cout << std::endl;

    // Get the platform
    status = clGetPlatformIDs(1, &platform, NULL);
    // Get the device
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 2, device, NULL);
    // Get the device name
    size_t device_name_size;
    clGetDeviceInfo(device[d], CL_DEVICE_NAME, 0, NULL, &device_name_size);
    char *device_name = new char[device_name_size];
    clGetDeviceInfo(device[d], CL_DEVICE_NAME, device_name_size, device_name, NULL);
    std::cout << "Using OpenCL device: " << device_name << std::endl;
    delete[] device_name;

    // Create the context
    context = clCreateContext(NULL, 1, &device[d], NULL, NULL, &status);

    // Create the program
    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, &status);
    // Build the program
    status = clBuildProgram(program, 1, &device[d], NULL, NULL, NULL);

    // Create the kernel
    kernel = clCreateKernel(program, "multiply_arrays", &status);

    // Create the command queue
    cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    queue = clCreateCommandQueueWithProperties(context, device[d], properties, &err);

    // Create the buffers
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &status);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), NULL, &status);
    bufferC = clCreateBuffer(context, CL_MEM_READ_WRITE, n * sizeof(float), NULL, &status);

    start_time = std::chrono::high_resolution_clock::now();
    // Copy data to the buffers
    status = clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, n * sizeof(float), a, 0, NULL, NULL);
    status = clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, n * sizeof(float), b, 0, NULL, NULL);
    // Set the kernel arguments
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&bufferA);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&bufferB);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&bufferC);

    // Execute the kernel
    size_t globalWorkSize[1] = {n};
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

    // Read the result from the buffer
    status = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, n * sizeof(float), c3, 0, NULL, NULL);
    end_time = std::chrono::high_resolution_clock::now();
    elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Method 3 Elapsed time: " << elapsed_time.count() << " ms" << std::endl;
    // print differences
    for (size_t i = 0; i < n; i++)
    {
        if (c1[i] != c3[i])
            std::cout << i << ",";
    }
    std::cout << std::endl;
    return 0;
}