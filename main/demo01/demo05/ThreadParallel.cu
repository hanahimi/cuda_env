/**
CUDA从入门到精通（五）：线程并行
在操作系统中，进程是资源分配的基本单元，而线程是CPU时间调度的基本单元（这里假设只有1个CPU）

将线程的概念引申到CUDA程序设计中，我们可以认为线程就是执行CUDA程序的最小单元
在GPU上每个线程都会运行一次该核函数
但GPU上的线程调度方式与CPU有很大不同。
CPU上会有优先级分配，从高到低，同样优先级的可以采用时间片轮转法实现线程调度。
GPU上线程没有优先级概念，所有线程机会均等，线程状态只有等待资源和执行两种状态，
如果资源未就绪，那么就等待；一旦就绪，立即执行。
当GPU资源很充裕时，所有线程都是并发执行的，这样加速效果很接近理论加速比；
而GPU资源少于总线程个数时，有一部分线程就会等待前面执行的线程释放资源，从而变为串行化执行。
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
	/**
	这5个线程是靠threadIdx这个内置变量知道自己的身份，它是个dim3类型变量，
	接受<<<>>>中第二个参数，它包含x,y,z 3维坐标，而我们传入的参数只有一维，所以只有x值是有效的。
	“addKernel<<<1, size>>>(dev_c, dev_a, dev_b);”
	通过核函数中int i = threadIdx.x;这一句，每个线程可以获得自身的id号，从而找到自己的任务去执行
	*/

    int i = threadIdx.x;
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

	// 运行核函数
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);
	/**
	<<<>>>表示运行时配置符号,里面1表示只分配一个线程组（又称线程块、Block），size表示每个线程组有size个线程（Thread）
	本程序中size根据前面传递参数个数应该为5，所以运行的时候，核函数在5个GPU线程单元上分别运行了一次，总共运行了5次。
	*/
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
	cudaStatus = cudaThreadSynchronize();   //同步线程
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
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
