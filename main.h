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
cl_platform_id platform;
cl_device_id device[3];
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel;
cl_mem bufferA, bufferB, bufferC;
cl_int status;
cl_int err;
cl_uint num_devices;
int d = 1; // OpenCL device 0 1st device ,1 2nd device