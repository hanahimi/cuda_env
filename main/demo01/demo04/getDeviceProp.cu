/**
������豸����ʶ
�������ǵĳ����Զ�ͨ������cuda API��������豸��Ŀ������
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
ʹ����cudaSetDevice(0)���������0��ʾ���������ĵ�һ���豸�ţ�
����ж���豸������Ϊ0,1,2...��

�ٿ����Ǳ�����ӵĴ��룬�и�����cudaGetDeviceCount(&num)���������������ȡ�豸������
��������ѡ������CUDA������豸��ȡֵ����0,1,...num-1�����ǿ���һ����ö���豸��
����cudaGetDeviceProperties(&prop)���������,Ȼ������һ������ɸѡ�㷨��
�ҵ����������Ӧ�õ��Ǹ��豸��opt��Ȼ�����cudaSetDevice(opt)����ѡ����豸��
ѡ���׼���ԴӴ����������汾���ơ����Ƶȸ����Ƕȳ�����
*/

