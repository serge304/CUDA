/* 
Sergey Kazachenko
Queen's University, the Department of Chemistry
Last modified: January 16, 2015
*/
   
/*! \file GPUArray.cuh
    \brief Defines the DeviceArray and DeviceHostArray classes
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
- destructor frees memory automatically; otherwise, if DeviceArray is global, clear() should be called
*/
template<typename T>
class DeviceArray
{
protected:
	std::size_t m_size;
	std::vector<T*> md_ptr;

private:
	// type definitions
	typedef DeviceArray<T>& reference;
	typedef const DeviceArray<T>& const_reference;

public:
	//! Constructor sets up number of GPUs and creates a list of Device pointers. The default number of Devices is 1. 
	DeviceArray(int ngpu = 1) : m_size(0)
	{
		int ndev;		
		checkCudaErrors(cudaGetDeviceCount(&ndev));
		if (ndev < ngpu)
		{
			std::stringstream msg;
			msg << "***ERROR!***" << ndev << " devices available, " << ngpu << " requested!" << std::endl;
			throw std::runtime_error(msg.str());			
		}
		md_ptr.resize(ngpu, 0);   // create a list of Device pointers
	}
	
	//! Copy constructor builds Device Array to be identical to src
	DeviceArray(const_reference src) 
	{
		md_ptr.resize(src.ngpu(), 0);
		resize(src.size());
		copy(src);	
	}

	//! Assignment operator rebuilds Device Array to be identical to rhs
	reference operator=(const_reference rhs)
	{
		if (this != &rhs) {
			if (ngpu() != rhs.ngpu()) {
				md_ptr.resize(rhs.ngpu(), 0);
				resize(rhs.size());
			} else if (size() != rhs.size())
				resize(rhs.size());
			copy(rhs);
		}
		return *this;
	}

	//! Copy content on all devices of input array to this array. Arrays must be the same size.  
	void copy(const_reference src)
	{
		if (this != &src) {
			if (bytes() != src.bytes())
				throw std::out_of_range("Copy error: arrays are not the same size!");
			for (int i = 0; i < ngpu() && i < src.ngpu(); i++)
				checkCudaErrors(cudaMemcpy(md_ptr[i], src.DevPtr(i), bytes(), cudaMemcpyDefault));
		}	
	}

	//! Changes the size of the array
	/*! 
		Set size. Delete old elements, if they exits. 
		Create new elements and set them to zero. 
	*/
	virtual void resize(std::size_t size)
	{
		m_size = size;
		for (int i = ngpu(); i--;) 
		{
			checkCudaErrors(cudaSetDevice(i));
			if (md_ptr[i]) checkCudaErrors(cudaFree(md_ptr[i]));
			checkCudaErrors(cudaMalloc((void**)&md_ptr[i], bytes()));
			checkCudaErrors(cudaMemset(md_ptr[i], 0, bytes()));
		}	
	}	

	//! Sets all elements to zero
	virtual void SetToZero()
	{
		for (int i = md_ptr.size(); i--;) {
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemset(md_ptr[i], 0, bytes()));
		}
	}

	//! A way to get an array size in elements 
	virtual std::size_t size() const {return m_size;}

	//! A way to get an array size in bytes 
	virtual std::size_t bytes() const {return sizeof(T)*m_size;}

	//! A way to get number of devices in use 
	virtual std::size_t ngpu() const {return md_ptr.size();}
	
	//! Returns a pointer to Device memory 
	virtual T* DevPtr(int gpu_id = 0) const {
		assert(gpu_id < ngpu()); 
		return md_ptr[gpu_id];
	}
	
	//! Copies data from Device zero to other available Devices
	virtual void UpdateSecondaryDevices()
	{
		for (int i = 1; i < ngpu(); i++)
			checkCudaErrors(cudaMemcpy(md_ptr[i], md_ptr[0], bytes(), cudaMemcpyDefault));
	}

	//! Erase all data from GPUarray and deallocate Device memory
	virtual void clear() 
	{
		for (int i = ngpu(); i--;)  
		{			
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaFree(md_ptr[i]));
		}
		m_size = 0;
		md_ptr.clear();
	}

	//! Destructor 
	virtual ~DeviceArray() {clear();}
};	

/*! \class DeviceHostArray 
    \brief simplifies data transfer between Host and Device.

	Resize and copy operations are done on Host. To copy data to device UpdateDevice() or copyUpdate() must be used.

This class adds a host vector to DeviceArray:
- several resize options
- copy between host and device 0
- [] operator to access host array
*/
template<typename T>
class DeviceHostArray : public DeviceArray<T>
{
protected:		
	std::vector<T> mh_data;

private:
	// type definitions
	typedef DeviceHostArray<T>& reference;
	typedef const DeviceHostArray<T>& const_reference;
	typedef DeviceArray<T> Base;

public:
	//! Constructor just sets number of devices
	DeviceHostArray(int ngpu = 1) : Base(ngpu) {};

	//! Copy constructor builds DeviceHostArray to be identical to src
	DeviceHostArray(const_reference src) : Base(src) {
		mh_data.resize(src.size(), src.VecRef());	
	}

	//! Assignment operator rebuilds DeviceHostArray to be identical to rhs
	reference operator=(const_reference rhs)
	{
		if (this != &rhs) {
			if (this->ngpu() != rhs.ngpu()) {
				this->md_ptr.resize(rhs.ngpu(), 0);
				resize(rhs.size(), rhs.VecRef());
			} else if (this->size() != rhs.size())
				resize(rhs.size(), rhs.VecRef());
			this->copy(rhs);
		}
		return *this;
	}

	//! Resizes Host and Device arrays 
	void resize(std::size_t size)
	{
		Base::resize(size);
		mh_data.resize(size);
	}
	
	//! Resizes Host and Device arrays, assigning a single value to Host array elements
	void resize(std::size_t size, T value)
	{
		Base::resize(size);
		mh_data.resize(size, value);
	}
	
	//! Resizes Host and Device arrays, copying values from the input vector to Host array
	void resize(std::size_t size, std::vector<T> &values)
	{		
		Base::resize(size);
		mh_data.resize(size);
		//
		assert(size <= values.size());
		for (int i = 0; i < size; i++)
			mh_data[i] = values[i];		
	}

	//! Copy host data from src to this array
	void copy(const_reference src)
	{
		if (this != &src) {
			assert(this->size() == src.size());
			for (int i = 0; i < this->size(); i++)
				mh_data[i] = src[i];			
		}	
	}

	//! Copy host data from src to this array and update devices
	void copyUpdate(const_reference src)
	{
		if (this != &src) {
			assert(Base::size() == src.size());
			copy(src);
			UpdateDevice();			
		}	
	}

	//! Copy device data from src to this array
	void copy(DeviceArray<T> src) {
		Base::copy(src);
	}

	//! Array subscript operator for the Host data 
	T& operator[](const std::size_t idx) {return mh_data.at(idx);}
	T& operator[](const std::size_t idx) const {return mh_data.at(idx);}

	//! Returns reference to the Host data 
	std::vector<T>& VecRef() {return mh_data;}
	std::vector<T>& VecRef() const {return mh_data;}

	//! Copies data from Host to all Devices 
	void UpdateDevice()
	{
		for (int i = this->ngpu(); i--;) {
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemcpy(Base::DevPtr(i), mh_data.data(), Base::bytes(), cudaMemcpyDefault));
		}
	}

	//! Copies data from Device 0 to Host 
	void UpdateHost() {checkCudaErrors(cudaMemcpy(mh_data.data(), Base::DevPtr(), Base::bytes(), cudaMemcpyDefault));}

	//! Erase all data from GPUarray and deallocate Device memory 
	void clear() 
	{
		Base::clear();
		mh_data.clear();		
	}

	//! Destructor 
	~DeviceHostArray() {clear();}
};

#endif
