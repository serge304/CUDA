#ifndef UTILS_H__
#define UTILS_H__

#include <stdio.h>
#include <iostream>

// Checks a return value from CUDA api functions. Adopted from Udacity CUDA cource.
#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)
template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(EXIT_FAILURE);
  }
}

// Calculates the number of blocks if the number of data points is not divisible by a selected number of threads
class Utils
{
public: 
	static int GetBlocks(int nX, int nT) { 
		if (nX%nT == 0)
			return nX/nT;
		else
			return nX/nT + 1; }
};

#endif
