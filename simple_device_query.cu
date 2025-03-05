#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <cuda_runtime.h>

int main(int argc, char *agrv[])
{
       int iDev = 0;
       cudaDeviceProp iProp;
       cudaGetDevice(&iDev);
       cudaGetDeviceProperties(&iProp, iDev);

       printf("Device %d: %s\n", iDev, iProp.name);
       printf("Number of multiprocessors:  %d \n",
              iProp.multiProcessorCount);
       printf("Total amount of global memory: %4.2f GB \n",
              iProp.totalGlobalMem / (1024 * 1024 * 1024.0));
       printf("Total amount of constant memory:  %4.2f KB \n",
              iProp.totalConstMem / 1024.0);
       printf("Total amount of shared memory per block:  %4.2f KB\n",
              iProp.sharedMemPerBlock / 1024.0);

       printf("Total amount of shared memory per multiprocessor:  %4.2f KB \n",
              iProp.sharedMemPerMultiprocessor / 1024.0);
       printf("Total number of registers available per block: %d\n",
              iProp.regsPerBlock);
       printf("Totoal number of registers available per SM: %d\n",
              iProp.regsPerMultiprocessor);
       printf("Warp size:                                     %d\n",
              iProp.warpSize);
       printf("Maximum number of threads per block:           %d\n",
              iProp.maxThreadsPerBlock);
       printf("Maximum blocks Per MultiProcessor: %d\n",
              iProp.maxBlocksPerMultiProcessor);
       printf("Maximum number of threads per multiprocessor:  %d\n",
              iProp.maxThreadsPerMultiProcessor);
       printf("Maximum number of warps per multiprocessor:    %d\n",
              iProp.maxThreadsPerMultiProcessor / iProp.warpSize);

       printf("local L1 cache supported: %s \n",
              iProp.localL1CacheSupported ? "true" :"false");


       printf("l2 cache szie %d MB \n", 
              iProp.l2CacheSize / (1024 * 1024));

       printf("persisting l2 cache max size: %d MB \n",
              iProp.persistingL2CacheMaxSize / (1024 * 1024));

       printf("Device can possibly execute multiple kernels concurrently: %s \n", iProp.concurrentKernels ? "true" : "false");

       

       
       int major, minor;


       cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, iDev);
       cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, iDev);

       printf("Compute capability: %d %d \n", major, minor);


       return EXIT_SUCCESS;
}