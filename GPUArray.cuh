/* 
Sergey Kazachenko
Queen's University, the Department of Chemistry
Last modified: September 16, 2014
*/
   
/*! \file GPUArray.cuh
    \brief Defines the DeviceArray and HostDeviceArray classes
*/

//#define NDEBUG 

#ifndef GPUARRAYCUH
#define GPUARRAYCUH

#include <assert.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include "GPUutils.cuh"

/*! \class DeviceArray
    \brief DeviceArray makes it easier to allocate and control data on multiple Devices 

Suggested use:
- declare (constructor sets number of devices to use)
- resize to required size
- use DevPtr() method to pass device pointer to a kernel
- use other methods where appropriate
- clear at the end of the program
*/
template<typename T>
class DeviceArray
{
protected:
	std::size_t m_size;
	std::size_t m_bytes;
	std::vector<T*> m_d_ptr;

public:
	// type definitions
	typedef DeviceArray<T>& reference;
	typedef const DeviceArray<T>& const_reference;

	//! Constructor sets up number of GPUs and creates a list of Device pointers. The default number of Devices is 1. 
	DeviceArray(int ngpu = 1) : m_size(0), m_bytes(0)
	{
		int ndev;		
		checkCudaErrors(cudaGetDeviceCount(&ndev));
		if (ndev < ngpu)
		{
			std::stringstream msg;
			msg << "***ERROR!***" << ndev << " devices available, " << ngpu << " requested!" << std::endl;
			throw std::runtime_error(msg.str());			
		}
		m_d_ptr.resize(ngpu, 0);   // create a list of Device pointers
	}

	//! Changes the size of the array
	/*! 
		Set size. Delete old elements, if they exits. 
		Create new elements and set them to zero. 
	*/
	virtual void resize(std::size_t size)
	{
		m_bytes = sizeof(T)*size;
		m_size = size;
		for (int i = ngpu(); i--;) 
		{
			checkCudaErrors(cudaSetDevice(i));
			if (m_d_ptr[i]) checkCudaErrors(cudaFree(m_d_ptr[i]));
			checkCudaErrors(cudaMalloc((void**)&m_d_ptr[i], m_bytes));
			checkCudaErrors(cudaMemset(m_d_ptr[i], 0, m_bytes));
		}	
	}	

	//! Sets all elements to zero
	virtual void SetToZero()
	{
		for (int i = m_d_ptr.size(); i--;) {
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemset(m_d_ptr[i], 0, m_bytes));
		}
	}

	//! Assignment operator rewrites content of a Device Array 
	reference operator=(const_reference other)
	{
		if (this != &other) {
			if (ngpu() != other.ngpu()) {
				m_d_ptr.resize(other.ngpu(), 0);
				resize(other.size());
			} else if (size() != other.size())
				resize(other.size());
			for (int i = 0; i < ngpu(); i++)
				checkCudaErrors(cudaMemcpy(m_d_ptr[i], other.DevPtr(i), m_bytes, cudaMemcpyDefault));	
		}
		return *this;
	}

	//! Copy content of input array to the device. Arrays be the same size.  
	void copy(const_reference other)
	{
		if (this != &other) {
			assert(m_bytes == other.bytes());
			for (int i = 0; i < ngpu() && i < other.ngpu(); i++)
				checkCudaErrors(cudaMemcpy(m_d_ptr[i], other.DevPtr(i), m_bytes, cudaMemcpyDefault));
		}	
	}

	//! A way to get an array size in elements 
	std::size_t size() const {return m_size;}

	//! A way to get an array size in bytes 
	std::size_t bytes() const {return m_bytes;}

	//! A way to get number of devices in use 
	std::size_t ngpu() const {return m_d_ptr.size();}
	
	//! Returns a pointer to Device memory 
	T* DevPtr(int gpu_id = 0) const {
		assert(gpu_id < ngpu()); 
		return m_d_ptr[gpu_id];
	}
	
	//! Copies data from Device zero to other available Devices
	void UpdateSecondaryDevices()
	{
		for (int i = 1; i < ngpu(); i++)
			checkCudaErrors(cudaMemcpy(m_d_ptr[i], m_d_ptr[0], m_bytes, cudaMemcpyDefault));
	}

	//! Erase all data from GPUarray and deallocate Device memory
	virtual void clear() 
	{
		for (int i = ngpu(); i--;)  
		{			
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaFree(m_d_ptr[i]));
		}
		m_size = 0;
		m_bytes = 0;
		m_d_ptr.clear();
	}

	//! Destructor 
	~DeviceArray() {clear();}
};	

/*! \class DeviceHostArray 
    \brief simplifies data transfer between Host and Device 

This class adds a host vector to DeviceArray:
- several resize options
- copy between host and device 0
- [] operator to access host array
*/
template<typename T>
class DeviceHostArray : public DeviceArray<T>
{
protected:		
	std::vector<T> m_h_data;

public:
	//! Constructor 
	DeviceHostArray(int ngpu = 1) : DeviceArray<T>(ngpu) {};

	//! Resizes Host and Device arrays 
	void resize(std::size_t size)
	{
		DeviceArray<T>::resize(size);
		m_h_data.resize(size);
	}
	
	//! Resizes Host and Device arrays, assigning a value to Host array 
	void resize(std::size_t size, T value)
	{
		DeviceArray<T>::resize(size);
		m_h_data.resize(size, value);
	}
	
	//! Resizes Host and Device arrays, copying values from the input vector to Host array
	void resize(std::size_t size, std::vector<T> &values)
	{		
		DeviceArray<T>::resize(size);
		m_h_data.resize(size);
		//
		assert(m_h_data.size() <= values.size());
		for (int i = 0; i < m_h_data.size(); i++)
			m_h_data[i] = values[i];
	}

	//! Array subscript operator for the Host data 
	T& operator[](const std::size_t idx) {return m_h_data.at(idx);}

	//! Returns pointer to the Host data 
	std::vector<T>* VecPtr() {return &m_h_data;}

	//! Copies data from Host to all Devices 
	void UpdateDevice()
	{
		for (int i = DeviceArray<T>::ngpu(); i--;) {
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemcpy(DeviceArray<T>::DevPtr(i), m_h_data.data(), DeviceArray<T>::bytes(), cudaMemcpyDefault));
		}
	}

	//! Copies data from Device 0 to Host 
	void UpdateHost() {checkCudaErrors(cudaMemcpy(m_h_data.data(), DeviceArray<T>::DevPtr(), DeviceArray<T>::bytes(), cudaMemcpyDefault));}

	//! Erase all data from GPUarray and deallocate Device memory 
	void clear() 
	{
		DeviceArray<T>::clear();
		m_h_data.clear();		
	}

	//! Destructor 
	~DeviceHostArray() {clear();}
};

#endif
