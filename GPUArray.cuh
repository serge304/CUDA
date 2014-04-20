/* 
Sergey Kazachenko
Queen's University, the Department of Chemistry
Last modified: April 10, 2014
*/
   
/*! \file GPUArray.cuh
    \brief Defines the DeviceArray and DeviceHostArray classes
*/

#define NDEBUG 

#ifndef GPUARRAYCUH
#define GPUARRAYCUH

#include <assert.h>
#include <iostream>
#include <vector>
#include "GPUutils.cuh"

/*! @brief DeviceArray makes it easier to allocate and control data on multiple Devices */
template<typename T>
class DeviceArray
{
protected:
	std::size_t m_size;
	std::vector<T*> m_d_ptr;

public:

	/*! @brief Constructor sets up number of GPUs and creates a list of Device pointers. The default nuber of Devices is 1. */
	DeviceArray(int ngpu = 1) : m_size(0)
	{
		int ndev;		
		checkCudaErrors(cudaGetDeviceCount(&ndev));
		if (ndev < ngpu)
			std::cerr << std::endl << "***WARNING!***" << ndev << " devices available, " << ngpu << " requested!" << std::endl;		
		m_d_ptr.resize(ngpu, 0);   // create a list of Device pointers
	}
	
	/*! @brief Changes size of the array, deleting old elements, if they exits.
	           Creates new elements and sets them to zero. */
	void resize(std::size_t size)
	{
		m_size = size;		
		for (int i = m_d_ptr.size(); i--;) 
		{
			checkCudaErrors(cudaSetDevice(i));
			if (m_d_ptr[i]) checkCudaErrors(cudaFree(m_d_ptr[i]));
			checkCudaErrors(cudaMalloc((void**)&m_d_ptr[i], sizeof(T)*m_size));
			checkCudaErrors(cudaMemset(m_d_ptr[i], 0, sizeof(T)*m_size));
		}
	}	

	//! @brief Sets all elements to zero
	void SetToZero()
	{
		for (int i = m_d_ptr.size(); i--;) {
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemset(m_d_ptr[i], 0, sizeof(T)*m_size));
		}
	}
	
	/*! @brief A way to get an array size */
	std::size_t size() {return m_size;}

	/*! @brief A way to get number of devices in use */
	std::size_t ngpu() {return m_d_ptr.size();}
	
	/*! @brief Returns a pointer to Device memory */
	T* DevPtr(int gpu_id = 0) {
		assert(gpu_id < m_d_ptr.size()); 
		return m_d_ptr[gpu_id];
	}
	
	/*! @brief Copies data from Device zero to other available Devices */
	void UpdateSecondaryDevices()
	{
		for (int i = 1; i < m_d_ptr.size(); i++)
			checkCudaErrors(cudaMemcpyPeer(m_d_ptr[i], i, m_d_ptr[0], 0, sizeof(T)*m_size));		
	}

	/*! @brief Erase all data from GPUarray and deallocate Device memory */
	void clear() 
	{
		for (int i = m_d_ptr.size(); i--;)  
		{			
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaFree(m_d_ptr[i]));
		}
		m_size = 0;
		m_d_ptr.clear();
	}

	/*! @brief Destructor */
	~DeviceArray() {clear();}
};	

/** @brief DeviceHostArray expands on DeviceArray to simplify data transfer between Host and Device */
template<typename T>
class DeviceHostArray : public DeviceArray<T>
{
protected:		
	std::vector<T> m_h_data;

public:

	/*! @brief Constructor */
	DeviceHostArray(int ngpu = 1) : DeviceArray<T>(ngpu) {};

	/*! @brief Resizes Host and Device arrays */
	void resize(std::size_t size)
	{
		DeviceArray<T>::resize(size);
		m_h_data.resize(this->m_size);
	}
	
	/*! @brief Resizes Host and Device arrays, assigning a value to Host array */
	void resize(std::size_t size, T value)
	{
		DeviceArray<T>::resize(size);
		m_h_data.resize(this->m_size, value);
	}
	
	/*! @brief Resizes Host and Device arrays, copying values from the input vector to Host array */
	void resize(std::size_t size, std::vector<T> *values)
	{		
		DeviceArray<T>::resize(size);
		m_h_data.resize(this->m_size);
		//
		assert(m_h_data.size() <= values->size());
		for (int i = 0; i < m_h_data.size(); i++)
			m_h_data[i] = values->at(i);
	}

	/*! @brief Array subscript operator for the Host data */
	T& operator[](const std::size_t idx) {return m_h_data.at(idx);}

	/*! @brief Returns pointer to the Host data */
	std::vector<T>* VecPtr() {return &m_h_data;}

	/*! @brief Copies data from Host to all Devices */
	void UpdateDevice()
	{
		for (int i = this->m_d_ptr.size(); i--;) {
			checkCudaErrors(cudaSetDevice(i));
			checkCudaErrors(cudaMemcpy(this->m_d_ptr[i], m_h_data.data(), sizeof(T)*this->m_size, cudaMemcpyHostToDevice));
		}
	}

	/*! @brief Copies data from Device 0 to Host */
	void UpdateHost() {checkCudaErrors(cudaMemcpy(m_h_data.data(), this->m_d_ptr[0], sizeof(T)*this->m_size, cudaMemcpyDeviceToHost));}

	/*! @brief Erase all data from GPUarray and deallocate Device memory */
	void clear() 
	{
		DeviceArray<T>::clear();
		m_h_data.clear();		
	}

	/*! @brief Destructor */
	~DeviceHostArray() {clear();}
};

#endif
