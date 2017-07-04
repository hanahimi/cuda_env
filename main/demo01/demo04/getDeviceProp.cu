/**
加深对设备的认识
是让我们的程序自动通过调用cuda API函数获得设备数目和属性
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "driver_types.h"
#include <stdio.h>

int main()
{
	cudaError_t cudaStatus;
	int num = 0;
	cudaDeviceProp prop;
	cudaStatus = cudaGetDeviceCount(&num);
	
	// Choose which GPU to run on, change this on a multi-GPU system.
	//cudaStatus = cudaSetDevice(0);
	
	for (int i = 0; i < num; i++)
	{
		cudaGetDeviceProperties(&prop, i);
	}

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;

}
/**
使用了cudaSetDevice(0)这个操作，0表示能搜索到的第一个设备号，
如果有多个设备，则编号为0,1,2...。

再看我们本节添加的代码，有个函数cudaGetDeviceCount(&num)，这个函数用来获取设备总数，
这样我们选择运行CUDA程序的设备号取值就是0,1,...num-1，于是可以一个个枚举设备，
利用cudaGetDeviceProperties(&prop)获得其属性,然后利用一定排序、筛选算法，
找到最符合我们应用的那个设备号opt，然后调用cudaSetDevice(opt)即可选择该设备。
选择标准可以从处理能力、版本控制、名称等各个角度出发。
*/

