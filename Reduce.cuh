/* 
Sergey Kazachenko
Queen's University, the Department of Chemistry
Last modified: November 19, 2014
*/

/*! \file Reduce.cuh
    Custom Reduce for arrays up to 512**3 elements. 
	The goal of custom code is to have intermediate arrays allocated only once, removing overhead for memory manipulation.
*/

#ifndef REDUCE_CUH
#define REDUCE_CUH

#include <vector>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "GPUutils.cuh"
#include "GPUArray.cuh"

namespace MyReduce
{	
	const int nElem = 9;
	const int nThreads  = 256;
	const int nThreads2 = 2*nThreads;
	const int nThreads2sq = nThreads2*nThreads2;
	const int nThreads2cub = nThreads2sq*nThreads2;

	//! @brief Intermediate arrays for types up to double9
	__device__ double d_Buffer2[nElem*nThreads2sq];  
	__device__ double d_Buffer1[nElem*nThreads2];   
	__device__ double d_Buffer0[nElem];             

	//! @brief Add operation
	template<typename T>
	class Add {
	public:
		__device__  __host__ T operator() (T &a, T &b) const { return a + b; }
	};

	//! @brief Max operation
	template<typename T>
	class Max {
	public:
		__device__   __host__ T operator() (T &a, T &b) const { return (a>b)?a:b; }
	};

	//! @brief Simple reduce kernel
	template<typename T, class O>
	__global__ void reduce_kernel(T *d_idata, T *d_odata, int n, O op)
	{
		__shared__ T sdata[nThreads];		
		int tid = threadIdx.x;
		int myId = blockIdx.x*nThreads2 + tid;		

		if (myId < n) // limit active threads by number of elements
		{		
			// each thread loads two elements from global to shared mem
			if (myId + nThreads < n)
				sdata[tid] = op(d_idata[myId], d_idata[myId + nThreads]); 
			else
		    // there are no second elements for the last ones
				sdata[tid] = d_idata[myId];
			__syncthreads();

			// reduction in shared mem
			for (unsigned int s = nThreads/2; s > 0; s >>= 1) 
			{
				if (tid < s && myId + s < n) 
					sdata[tid] = op(sdata[tid], sdata[tid + s]);
				__syncthreads();
			}
			// write result for this block to global mem
			if (tid == 0)
				d_odata[blockIdx.x] = sdata[0];
		}
	}

	//! @brief kernel driver on the host side
	template<typename T, class O>
	T Reduce(DeviceArray<T>& d_in, int n, O op, int ngpu = 1)
	{
		T val; // = T(0);
		memset((void*)&val, 0, sizeof(T));
		std::vector<T*> ptrBuffer2(ngpu), ptrBuffer1(ngpu), ptrBuffer0(ngpu);

		// range check
		if (n < 2)
			throw std::out_of_range("Reduce error: array size is too small");
		if (n > nThreads2cub)
			throw std::out_of_range("Reduce error: array size is too large");		
		//
		for (int igpu = ngpu; igpu--;) {
			checkCudaErrors(cudaSetDevice(igpu));
			checkCudaErrors(cudaGetSymbolAddress((void**)&ptrBuffer2[igpu], d_Buffer2));
			checkCudaErrors(cudaGetSymbolAddress((void**)&ptrBuffer1[igpu], d_Buffer1));
			checkCudaErrors(cudaGetSymbolAddress((void**)&ptrBuffer0[igpu], d_Buffer0));			
		}
		 
		// First step.
		// Each block reduces nThreads2 elements
		if (n > nThreads2sq)
		{
			int nBlocks = Utils::GetBlocks(n, nThreads2); 
			for (int igpu = ngpu; igpu--;) {
				checkCudaErrors(cudaSetDevice(igpu));
				reduce_kernel<<<nBlocks, nThreads>>>(d_in.DevPtr(igpu), ptrBuffer2[igpu], n, op); 	
			}
			n = nBlocks;
		}
		else
		{
			for (int igpu = ngpu; igpu--;) 
				ptrBuffer2[igpu] = d_in.DevPtr(igpu);
		}

		// Second step. 
		// Each block reduces nThreads2 elements
		if (n > nThreads2)
		{
			int nBlocks = Utils::GetBlocks(n, nThreads2); 
			for (int igpu = ngpu; igpu--;) {
				checkCudaErrors(cudaSetDevice(igpu));
				reduce_kernel<<<nBlocks, nThreads>>>(ptrBuffer2[igpu], ptrBuffer1[igpu], n, op); 	
			}
			n = nBlocks;
		}
		else
		{
			for (int igpu = ngpu; igpu--;) 
				ptrBuffer1[igpu] = ptrBuffer2[igpu];
		}

		// Third step. 
		// One block reduces up to nThreads2 elements
		for (int igpu = ngpu; igpu--;) {
			checkCudaErrors(cudaSetDevice(igpu));
			reduce_kernel<<<1, nThreads>>>(ptrBuffer1[igpu], ptrBuffer0[igpu], n, op);
		}

		// get values to the host and combine
		T tmp;
		for (int igpu = ngpu; igpu--;)	{			
			checkCudaErrors(cudaSetDevice(igpu));
			checkCudaErrors(cudaMemcpy(&tmp, ptrBuffer0[igpu], sizeof(T), cudaMemcpyDefault));
			val = op(val, tmp);
		}

		return val;
	}

	// Sum version
	template<typename T>
	T Sum(DeviceArray<T>& d_in, int n, int ngpu = 1) {
		return Reduce(d_in, n, Add<T>(), ngpu);
	}

	template<typename T>
	T Sum(DeviceArray<T>& d_in) {
		return Reduce(d_in, d_in.size(), Add<T>(), d_in.ngpu());
	}

	// Max version
	template<typename T>
	T Maximum (DeviceArray<T>& d_in, int n, int ngpu = 1) {
		return Reduce(d_in, n, Max<T>(), ngpu);
	}

	template<typename T>
	T Maximum (DeviceArray<T>& d_in) {
		return Reduce(d_in, d_in.size(), Max<T>(), d_in.ngpu());
	}
}

#endif
