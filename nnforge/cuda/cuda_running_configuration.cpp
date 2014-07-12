/*
 *  Copyright 2011-2014 Maxim Milakov
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include "cuda_running_configuration.h"

#include <ostream>
#include <boost/format.hpp>
#include <limits>

#include <cuda.h>
#include <cuda_runtime.h>

#include "neural_network_cuda_exception.h"
#include "neural_network_cublas_exception.h"


namespace nnforge
{
	namespace cuda
	{
		cuda_running_configuration::cuda_running_configuration(
			int device_id,
			float max_global_memory_usage_ratio)
			: device_id(device_id)
			, max_global_memory_usage_ratio(max_global_memory_usage_ratio)
			, cublas_handle(0)
		{
			update_parameters();
		}

		cuda_running_configuration::~cuda_running_configuration()
		{
			if (cublas_handle)
				cublasDestroy(cublas_handle);
			cudaDeviceReset();
		}

		void cuda_running_configuration::update_parameters()
		{
	        cuda_safe_call(cudaDriverGetVersion(&driver_version));
	        cuda_safe_call(cudaRuntimeGetVersion(&runtime_version));

			int device_count;
		    cuda_safe_call(cudaGetDeviceCount(&device_count));
			if (device_count <= 0)
				throw neural_network_exception("No CUDA capable devices are found");

			if (device_id >= device_count)
				throw neural_network_exception((boost::format("Device ID %1% specified while %2% devices are available") % device_id % device_count).str());

			cudaDeviceProp device_prop;
			cuda_safe_call(cudaGetDeviceProperties(&device_prop, device_id));
			device_name = device_prop.name;
			compute_capability_major = device_prop.major;
			compute_capability_minor = device_prop.minor;
			clock_rate = device_prop.clockRate;
			memory_clock_rate = device_prop.memoryClockRate;
			memory_bus_width = device_prop.memoryBusWidth;
			global_memory_size = device_prop.totalGlobalMem;
			ecc_enabled = (device_prop.ECCEnabled != 0);
			l2_cache_size = device_prop.l2CacheSize;
			multiprocessor_count = device_prop.multiProcessorCount;
			smem_per_block = device_prop.sharedMemPerBlock;
			max_threads_per_multiprocessor = device_prop.maxThreadsPerMultiProcessor;
			max_threads_per_block = device_prop.maxThreadsPerBlock;
			for(int i = 0; i < sizeof(max_threads_dim) / sizeof(max_threads_dim[0]); ++i)
				max_threads_dim[i] = device_prop.maxThreadsDim[i];
			for(int i = 0; i < sizeof(max_grid_size) / sizeof(max_grid_size[0]); ++i)
				max_grid_size[i] = device_prop.maxGridSize[i];
			max_texture_1d_linear = device_prop.maxTexture1DLinear;
			texture_alignment = device_prop.textureAlignment;
			pci_bus_id = device_prop.pciBusID;
			pci_device_id = device_prop.pciDeviceID;
		#ifdef _WIN32
			tcc_mode = (device_prop.tccDriver != 0);
		#endif

			cuda_safe_call(cudaSetDevice(device_id));

			cublas_safe_call(cublasCreate(&cublas_handle));
		}

		bool cuda_running_configuration::is_flush_required() const
		{
			#ifdef _WIN32
				return !tcc_mode;
			#else
				return false;
			#endif
		}

		std::ostream& operator<< (std::ostream& out, const cuda_running_configuration& running_configuration)
		{
			out << "--- CUDA versions ---" << std::endl;
			out << "Driver version = " << running_configuration.driver_version / 1000 << "." << (running_configuration.driver_version % 100) / 10 << std::endl;
			out << "Runtime version = " << running_configuration.runtime_version / 1000 << "." << (running_configuration.runtime_version % 100) / 10 << std::endl;

			out << "--- Device ---" << std::endl;

			out << "Device Id = " << running_configuration.device_id << std::endl;
			out << "Device name = " << running_configuration.device_name << std::endl;
			out << "Compute capability = " << running_configuration.compute_capability_major << "." << running_configuration.compute_capability_minor << std::endl;
			out << "Clock rate = " << (running_configuration.clock_rate / 1000) << " MHz" << std::endl;
			out << "Memory clock rate = " << (running_configuration.memory_clock_rate / 1000) << " MHz" << std::endl;
			out << "Memory bus width = " << running_configuration.memory_bus_width << " bits" << std::endl;
			out << "Global memory size = " << running_configuration.global_memory_size / (1024 * 1024) << " MB" << std::endl;
			out << "ECC support = " << (running_configuration.ecc_enabled ? "Enabled" : "Disabled") << std::endl;
			out << "L2 cache size = " << running_configuration.l2_cache_size << " bytes" << std::endl;
			out << "Multiprocessor count = " << running_configuration.multiprocessor_count << std::endl;
			out << "Shared memory per block size = " << running_configuration.smem_per_block << " bytes" << std::endl;
			out << "Maximum number of threads per multiprocessor = " << running_configuration.max_threads_per_multiprocessor << std::endl;
			out << "Maximum number of threads per block = " << running_configuration.max_threads_per_block << std::endl;
			out << "Maximum sizes of each dimension of a block = "
				<< running_configuration.max_threads_dim[0] << " x "
				<< running_configuration.max_threads_dim[1] << " x "
				<< running_configuration.max_threads_dim[2] << std::endl;
			out << "Maximum sizes of each dimension of a grid = "
				<< running_configuration.max_grid_size[0] << " x "
				<< running_configuration.max_grid_size[1] << " x "
				<< running_configuration.max_grid_size[2] << std::endl;
			out << "Maximum size of 1D texture bound to linear memory = " << running_configuration.max_texture_1d_linear << std::endl;
			out << "Texture alignment = " << running_configuration.texture_alignment << " bytes" << std::endl;
			out << "PCI Bus ID = " << running_configuration.pci_bus_id << std::endl;
			out << "PCI Location ID = " << running_configuration.pci_device_id << std::endl;
			#ifdef WIN32
				out << "Driver mode = " << (running_configuration.tcc_mode ? "TCC" : "WDDM") << std::endl;
			#endif

			out << "--- Settings ---" << std::endl;

			out << "Max global memory usage ratio = " << running_configuration.max_global_memory_usage_ratio << std::endl;

			out << "--- Status ---" << std::endl;

			size_t free_memory;
			size_t total_memory;
			cuda_safe_call(cudaMemGetInfo(&free_memory, &total_memory));

			out << "Free memory = " << free_memory / (1024 * 1024) << " MB" << std::endl;
			out << "Total memory = " << total_memory / (1024 * 1024) << " MB" << std::endl;

			return out;
		}

		unsigned int cuda_running_configuration::get_max_entry_count(
			const buffer_cuda_size_configuration& buffers_config,
			float ratio) const
		{
			size_t memory_left = static_cast<size_t>(static_cast<float>(global_memory_size) * max_global_memory_usage_ratio * ratio) - buffers_config.constant_buffer_size;
			if (memory_left< 0)
				memory_left = 0;

			size_t entry_count_limited_by_global = memory_left / buffers_config.per_entry_buffer_size;

			unsigned int entry_count_limited_by_linear_texture = buffers_config.max_tex_per_entry > 0 ? (max_texture_1d_linear - 1) / buffers_config.max_tex_per_entry : std::numeric_limits<int>::max();

			unsigned int entry_count = std::min<unsigned int>(entry_count_limited_by_global, entry_count_limited_by_linear_texture);

			return entry_count;
		}

		cublasHandle_t cuda_running_configuration::get_cublas_handle() const
		{
			return cublas_handle;
		}
	}
}
