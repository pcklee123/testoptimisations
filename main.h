#include <iostream>
#include <fstream>
#include <chrono>
#include <cstdlib>     // for rand()
#include <immintrin.h> // Include AVX-512 intrinsics
#include <emmintrin.h> // include header for SSE instructions
#include <omp.h>       // Include OpenMP
#include <stdio.h>
#include <stdlib.h>
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

// OpenCL variables
#define max_cl_dev 3
//cl_platform_id platform[max_cl_dev];
cl_platform_id platform;
cl_device_id device[max_cl_dev];
cl_context context[max_cl_dev];
cl_command_queue queue[max_cl_dev];
cl_program program[max_cl_dev];
cl_kernel kernel1[max_cl_dev];
cl_kernel kernel2[max_cl_dev];
cl_kernel kernel3[max_cl_dev];
cl_mem bufferA[max_cl_dev], bufferB[max_cl_dev], bufferC[max_cl_dev];
//cl_int status[max_cl_dev];
//cl_int err[max_cl_dev];
cl_int status1;
cl_uint num_devices;
