/**
CUDA�����ŵ���ͨ���������鲢��
�鲢���൱�ڲ���ϵͳ�ж���̵�������Ͻ�˵����CUDA���߳��飨�߳̿飩�ĸ��
��һ���߳���֯��һ�𣬹�ͬ����һ������Դ��Ȼ���ڲ�����ִ�С�
�߳̿����߳̿�֮�䣬���޹ϸ������������������ȵĲ��С�
���ǽ���һ�ڵĴ����Ϊ�鲢�а汾���£�
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

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<size, 1>>>(dev_c, dev_a, dev_b);
	/**
	����һ����ȣ�ֻ���������иı䣬<<<>>>���һ�������ĳ���size���ڶ����ĳ���1��
	��ʾ���Ƿ���size���߳̿飬ÿ���߳̿������1���̣߳��ܹ�������5���̡߳�
	��5���߳��໥������ִ�к˺����õ���Ӧ�Ľ����
	����һ�ڲ�ͬ���ǣ�ÿ���̻߳�ȡid�ķ�ʽ��Ϊint i = blockIdx.x�������߳̿�ID
	*/
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
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
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}


/**
�̲߳��кͿ鲢�е����������

�̲߳�����ϸ���Ȳ��У�����Ч�ʸߣ�
�鲢���Ǵ����Ȳ��У�ÿ�ε��ȶ�Ҫ���·�����Դ��
��ʱ��Դֻ��һ�ݣ���ô�����߳̿鶼ֻ���ų�һ�ӣ�����ִ�С�

���ǲ�����������ʱ��Ӧ�����̲߳��У������ܲ��ÿ鲢�У�

��Ȼ���ǣ����ǵ�������ʱ���Բ��÷��η�����һ��������ֽ�Ϊ����С��ģ���⣬
����ЩС��ģ����ֱ���һ���߳̿�ʵ��
�߳̿��ڿ��Բ���ϸ���ȵ��̲߳��У�����֮��Ϊ�����Ȳ��У��������Գ������Ӳ����Դ�������̲߳��еļ��㸴�Ӷ�
�ʵ��ֽ⣬���͹�ģ����һЩ����˷��������ڻ�����Ӧ���п��Եõ���ֵ�չʾ��
*/