/* 
*  Sergey Kazachenko
*  Queen's University, the Department of Chemistry
*  Last modified: April 10, 2014
*/

/*! \file Reduce.cu
    Custom Reduce for arrays up to nThreads2*nThreads2 elements. 
	The goal of custom code is to have intermediate arrays allocated only once, removing overhead for memory manipulation in Thrust reduce.
	Thrust library functions are used for cases that are not supported by the custom Reduce.
*/

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "GPUutils.cuh"
#include "Reduce.cuh"

namespace Reduce
{
	const int nThreads  = 256;
	const int nThreads2 = 2*nThreads;

	//! @brief Intermediate arrays
	__device__ double d_buffer[9*nThreads2];  // intermediate array of size nThreads2 elemets for types up to double9
	__device__ double d_out[9];               // output array for types up to double9;

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

	//! @brief Thrust sum reduce
	template<typename T>
	double SumReduceThrust(T *d_array, int n)
	{
		thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(d_array);
		return thrust::reduce(d_ptr, d_ptr + n, (T)0, thrust::plus<T>());
	}

	//! @brief Thrust max reduce
	template<typename T>
	double MaxReduceThrust(T *d_array, int n)
	{
		thrust::device_ptr<T> d_ptr = thrust::device_pointer_cast(d_array);
		return (thrust::reduce(d_ptr, d_ptr + n, (T)0, thrust::maximum<T>()));
	}

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
	T Reduce (T *in, int n, O op)
	{
		T val;
		T *ptrBuffer, *ptrOut;

		// range check
		if (n < 2)
			throw std::out_of_range("Incorrect array size for Reduce");
		
		//
		checkCudaErrors(cudaGetSymbolAddress((void**)&ptrBuffer, d_buffer));
		checkCudaErrors(cudaGetSymbolAddress((void**)&ptrOut, d_out));

		// First step. 
		// Each block reduces nThreads2 elements
		if (n > nThreads2)
		{
			int nBlocks = Utils::GetBlocks(n, nThreads2); 
			reduce_kernel<<<nBlocks, nThreads>>>(in, ptrBuffer, n, op); 	
			n = nBlocks;
		}
		else
			ptrBuffer = in;

		// Second step. 
		// One block reduces up to nThreads2 elements
		reduce_kernel<<<1, nThreads>>>(ptrBuffer, ptrOut, n, op);
		checkCudaErrors(cudaMemcpy(&val, ptrOut, sizeof(T), cudaMemcpyDefault));

		//
		return val;
	}

	//! @brief  Sum-reduce 
	// if data size is large use library function
	template<typename T>
	T Sum_driver(T *d_in, int n)
	{		
		if (n > nThreads2*nThreads2) 
			return SumReduceThrust(d_in, n);
		else
			return Reduce(d_in, n, Add<T>()); 		
	}

	//! @brief  Max-reduce 
	// if data size is large use library function
	template<typename T>
	T Max_driver(T *d_in, int n)
	{
		if (n > nThreads2*nThreads2) 
			return MaxReduceThrust(d_in, n);
		else
			return Reduce(d_in, n, Max<T>()); 		
	}

	//! @brief  Sum-reduce for double precision 	
	int Sum(int *d_in, int n) { return Sum_driver(d_in, n); }	
	//! @brief  Sum-reduce for integer 
	double Sum(double *d_in, int n) { return Sum_driver(d_in, n); }

	//! @brief Max-reduce for double precision
	double Maximum(double *d_in, int n) { return Max_driver(d_in, n); }
	//! @brief Max-reduce for integer
	int Maximum(int *d_in, int n) { return Max_driver(d_in, n); }

}
