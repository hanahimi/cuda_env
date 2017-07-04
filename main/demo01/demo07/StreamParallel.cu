/**
CUDA从入门到精通（七）：流并行

前面已经介绍了线程并行和块并行，知道了线程并行为细粒度的并行，而块并行为粗粒度的并行，
同时也知道了CUDA的线程组织情况，即Grid-Block-Thread结构。
一组线程并行处理可以组织为一个block，而一组block并行处理可以组织为一个Grid，
很自然地想到，Grid只是一个网格，我们是否可以利用多个网格来完成并行处理呢？答案就是利用流。

流可以实现在一个设备上运行多个核函数。
前面的块并行也好，线程并行也好，运行的核函数都是相同的（代码一样，传递参数也一样）。
而流并行，可以执行不同的核函数，也可以实现对同一个核函数传递不同的参数，实现任务级别的并行。
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = blockIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
	// 创建流
	cudaStream_t stream[5];
	for (int i = 0; i < 5; i++)
	{
		cudaStreamCreate(&stream[i]);
	}
    // 执行流
	for (int i = 0; i < 5; i++)
	{	
		// Launch a kernel on the GPU with one thread for each element.
		addKernel << <1, 1, 0, stream[i] >> >(dev_c + i, dev_a + i, dev_b + i);
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
	// 销毁流
	for (int i = 0; i < 5; i++)
	{
		cudaStreamDestroy(stream[i]);
	}
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

/**
注意到，我们的核函数代码仍然和块并行的版本一样，只是在调用时做了改变，<<<>>>中的参数多了两个
其中前两个和块并行、线程并行中的意义相同，仍然是线程块数（这里为1）、每个线程块中线程数（这里也是1）
第三个为0表示每个block用到的共享内存大小，这个我们后面再讲
第四个为流对象，表示当前核函数在哪个流上运行。
我们创建了5个流，每个流上都装载了一个核函数，同时传递参数有些不同，也就是每个核函数作用的对象也不同。
这样就实现了任务级别的并行，当我们有几个互不相关的任务时，可以写多个核函数，
资源允许的情况下，我们将这些核函数装载到不同流上，然后执行，这样可以实现更粗粒度的并行

*/