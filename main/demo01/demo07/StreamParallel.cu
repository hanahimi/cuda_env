/**
CUDA�����ŵ���ͨ���ߣ���������

ǰ���Ѿ��������̲߳��кͿ鲢�У�֪�����̲߳���Ϊϸ���ȵĲ��У����鲢��Ϊ�����ȵĲ��У�
ͬʱҲ֪����CUDA���߳���֯�������Grid-Block-Thread�ṹ��
һ���̲߳��д��������֯Ϊһ��block����һ��block���д��������֯Ϊһ��Grid��
����Ȼ���뵽��Gridֻ��һ�����������Ƿ�������ö����������ɲ��д����أ��𰸾�����������

������ʵ����һ���豸�����ж���˺�����
ǰ��Ŀ鲢��Ҳ�ã��̲߳���Ҳ�ã����еĺ˺���������ͬ�ģ�����һ�������ݲ���Ҳһ������
�������У�����ִ�в�ͬ�ĺ˺�����Ҳ����ʵ�ֶ�ͬһ���˺������ݲ�ͬ�Ĳ�����ʵ�����񼶱�Ĳ��С�
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
	// ������
	cudaStream_t stream[5];
	for (int i = 0; i < 5; i++)
	{
		cudaStreamCreate(&stream[i]);
	}
    // ִ����
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
	// ������
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
ע�⵽�����ǵĺ˺���������Ȼ�Ϳ鲢�еİ汾һ����ֻ���ڵ���ʱ���˸ı䣬<<<>>>�еĲ�����������
����ǰ�����Ϳ鲢�С��̲߳����е�������ͬ����Ȼ���߳̿���������Ϊ1����ÿ���߳̿����߳���������Ҳ��1��
������Ϊ0��ʾÿ��block�õ��Ĺ����ڴ��С��������Ǻ����ٽ�
���ĸ�Ϊ�����󣬱�ʾ��ǰ�˺������ĸ��������С�
���Ǵ�����5������ÿ�����϶�װ����һ���˺�����ͬʱ���ݲ�����Щ��ͬ��Ҳ����ÿ���˺������õĶ���Ҳ��ͬ��
������ʵ�������񼶱�Ĳ��У��������м���������ص�����ʱ������д����˺�����
��Դ���������£����ǽ���Щ�˺���װ�ص���ͬ���ϣ�Ȼ��ִ�У���������ʵ�ָ������ȵĲ���

*/