#include "main.h"
#define nn 4096
int main()
{
    const size_t n = 1024 * 1024;
    // allocate an array of n integers aligned to a 32-byte(256bit) boundary
    float *a = (float *)_mm_malloc(n * sizeof(float), 64);
    float *b = (float *)_mm_malloc(n * sizeof(float), 64);
    float *c1 = (float *)_mm_malloc(n * sizeof(float), 64);
    float *c2 = (float *)_mm_malloc(n * sizeof(float), 64);
    float *c3_ptr = (float *)_mm_malloc(max_cl_dev * n * sizeof(float), 64);
    float(*c3)[n] = (float(*)[n])c3_ptr;
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
        for (size_t j = 0; j < nn; j++)
            c1[i] *= (c1[i] + 1.0) * (a[i] + 1.0) * (b[i] + 1.0);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "Method 1 Elapsed time: " << elapsed_time.count() << " ms" << std::endl;

    // Method 2
    //  do all elements and store the result in c before going on to the next step of calculations
    start_time = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        c2[i] = a[i] * b[i];
#pragma omp barrier
    for (size_t j = 0; j < nn; j++)
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

    // method 3
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
        kernel[j] = clCreateKernel(program[j], "multiply_arrays", &status);
    }
#pragma omp barrier
//#pragma omp parallel for
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
        status = clSetKernelArg(kernel[j], 0, sizeof(cl_mem), (void *)&bufferA[j]);
        status = clSetKernelArg(kernel[j], 1, sizeof(cl_mem), (void *)&bufferB[j]);
        status = clSetKernelArg(kernel[j], 2, sizeof(cl_mem), (void *)&bufferC[j]);

        // Execute the kernel
        size_t globalWorkSize[1] = {n};
        status = clEnqueueNDRangeKernel(queue[j], kernel[j], 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

        // Read the result from the buffer
        status = clEnqueueReadBuffer(queue[j], bufferC[j], CL_TRUE, 0, n * sizeof(float), c3[j], 0, NULL, NULL);
        end_time = std::chrono::high_resolution_clock::now();
        elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "Method 3 Elapsed time: " << elapsed_time.count() << " ms thread" << j << ": Using OpenCL device: " << device_name[j] << std::endl;

        // print differences
        for (size_t i = 0; i < n; i++)
        {
            if (c1[i] != c3[j][i])
                std::cout << i << ",";
        }
        std::cout << std::endl;
    }
    return 0;
}