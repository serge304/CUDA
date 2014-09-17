#ifndef REDUCE_CUH
#define REDUCE_CUH

// Four types of reduce operation. Requires device pointer and an array size
namespace Reduce
{	
	double Sum(double*, int);
	int Sum(int*, int);
	double Maximum(double*, int);	
	int Maximum(int*, int);
}

#endif
