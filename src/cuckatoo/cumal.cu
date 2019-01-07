#include <stdio.h>
#include <inttypes.h>
#include <assert.h>

int main(int argc, char **argv) {
  size_t bufferMB;
  void *buffer;
  int device = argc > 1 ? atoi(argv[1]) : 1;
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  assert(device < nDevices);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  uint64_t dbytes = prop.totalGlobalMem;
  printf("%s with %d MB @ %d bits x %dMHz\n", prop.name, (uint32_t)(dbytes>>20), prop.memoryBusWidth, prop.memoryClockRate/1000);

  cudaSetDevice(device);
  for (bufferMB = 11 * 1024; ; bufferMB -= 16) {
    int ret = cudaMalloc((void**)&buffer, bufferMB << 20);
    if (ret) printf("cudaMalloc(%d MB) returned %d\n", bufferMB, ret);
    else break;
  }
  printf("cudaMalloc(%d MB) succeeded %d\n", bufferMB);
  cudaFree(buffer);

  return 0;
}
