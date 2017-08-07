/*
 *  Copyright 2011-2017 Maxim Milakov
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
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cudnn_util.h"

#include "neural_network_cuda_exception.h"
#include "neural_network_cublas_exception.h"
#include "neural_network_cusparse_exception.h"
#include "neural_network_cudnn_exception.h"
#include "neural_network_curand_exception.h"
#include "../rnd.h"

namespace nnforge
{
	namespace cuda
	{
		cuda_running_configuration::cuda_running_configuration(
			int device_id,
			float max_global_memory_usage_ratio,
			float cuda_fixed_working_buffers_ratio,
			int device_pos,
			cuda_communicator::ptr communicator)
			: device_id(device_id)
			, max_global_memory_usage_ratio(max_global_memory_usage_ratio)
			, cuda_fixed_working_buffers_ratio(cuda_fixed_working_buffers_ratio)
			, device_pos(device_pos)
			, communicator(communicator)
			, cublas_handle(0)
			, cusparse_handle(0)
			, cudnn_handle(0)
			, curand_gen(0)
		{
			update_parameters();
		}

		cuda_running_configuration::~cuda_running_configuration()
		{
			set_device();
			if (cublas_handle)
				cublasDestroy(cublas_handle);
			if (cusparse_handle)
				cusparseDestroy(cusparse_handle);
			if (cudnn_handle)
				cudnnDestroy(cudnn_handle);
			if (curand_gen)
				curandDestroyGenerator(curand_gen);
		}

		void cuda_running_configuration::update_parameters()
		{
			set_device();

			cuda_safe_call(cudaSetDeviceFlags(cudaDeviceScheduleYield | cudaDeviceMapHost));

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
			texture_alignment = static_cast<int>(device_prop.textureAlignment);
			pci_bus_id = device_prop.pciBusID;
			pci_device_id = device_prop.pciDeviceID;
		#ifdef _WIN32
			tcc_mode = (device_prop.tccDriver != 0);
		#endif

			cuda_safe_call(cudaDeviceGetStreamPriorityRange(&least_stream_priority, &greatest_stream_priority));

			if (compute_capability_major < 3)
			{
				throw neural_network_exception((boost::format("Insufficient compute capability %1%.%2% for device #%3% \"%4%\". Kepler and above architectures supported only.") % compute_capability_major % compute_capability_minor % device_id % device_name).str());
			}

			cublas_safe_call(cublasCreate(&cublas_handle));

			cusparse_safe_call(cusparseCreate(&cusparse_handle));

			cudnn_safe_call(cudnnCreate(&cudnn_handle));

			curand_safe_call(curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_MRG32K3A));
			curand_safe_call(curandSetPseudoRandomGeneratorSeed(curand_gen, rnd::get_time_dependent_seed()));
		}

		void cuda_running_configuration::set_device() const
		{
			cuda_safe_call(cudaSetDevice(device_id));
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
			running_configuration.set_device();

			out << "--- Device " << running_configuration.device_name << " ---" << std::endl;

			out << "Device Id = " << running_configuration.device_id << std::endl;
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
			#ifdef _WIN32
				out << "Driver mode = " << (running_configuration.tcc_mode ? "TCC" : "WDDM") << std::endl;
			#endif
			out << "Estimated GFLOPS = " << static_cast<int>(running_configuration.get_flops() / 1.0e+9F) << std::endl;
			out << "Stream priorities " << ((running_configuration.greatest_stream_priority != running_configuration.least_stream_priority) ? "present" : "absent") << std::endl;

			out << "Max global memory usage ratio = " << running_configuration.max_global_memory_usage_ratio << std::endl;
			out << "Fixed working buffers ratio = " << running_configuration.cuda_fixed_working_buffers_ratio << std::endl;

			size_t free_memory;
			size_t total_memory;
			cuda_safe_call(cudaMemGetInfo(&free_memory, &total_memory));

			out << "Free memory = " << free_memory / (1024 * 1024) << " MiB" << std::endl;
			out << "Total memory = " << total_memory / (1024 * 1024) << " MiB" << std::endl;

			return out;
		}

		unsigned int cuda_running_configuration::get_max_entry_count(
			const buffer_cuda_size_configuration& buffers_config,
			float ratio) const
		{
			long long memory_left = static_cast<long long>(static_cast<float>(global_memory_size) * max_global_memory_usage_ratio * ratio) - static_cast<long long>(buffers_config.constant_buffer_size);
			if (memory_left < 0)
				memory_left = 0;

			size_t entry_count_limited_by_global = memory_left / buffers_config.per_entry_buffer_size;

			unsigned int entry_count_limited_by_linear_texture = buffers_config.max_tex_per_entry > 0 ? (max_texture_1d_linear - 1) / buffers_config.max_tex_per_entry : std::numeric_limits<int>::max();

			unsigned int entry_count = std::min(static_cast<unsigned int>(entry_count_limited_by_global), entry_count_limited_by_linear_texture);

			return entry_count;
		}

		size_t cuda_running_configuration::get_max_fixed_working_buffers_size() const
		{
			return static_cast<size_t>(static_cast<float>(global_memory_size) * max_global_memory_usage_ratio * cuda_fixed_working_buffers_ratio);
		}

		cublasHandle_t cuda_running_configuration::get_cublas_handle() const
		{
			return cublas_handle;
		}

		cusparseHandle_t cuda_running_configuration::get_cusparse_handle() const
		{
			return cusparse_handle;
		}

		cudnnHandle_t cuda_running_configuration::get_cudnn_handle() const
		{
			return cudnn_handle;
		}

		curandGenerator_t cuda_running_configuration::get_curand_generator() const
		{
			return curand_gen;
		}

		int cuda_running_configuration::get_core_count_per_sm() const
		{
			if (compute_capability_major <= 3)
				return 192;
			else if (((compute_capability_major == 6) && (compute_capability_minor == 0)) || (compute_capability_major >= 7))
				return 64;
			else
				return 128;
		}

		float cuda_running_configuration::get_flops() const
		{
			return static_cast<float>(get_core_count_per_sm() * 2 * multiprocessor_count) * clock_rate * 1000.0F;
		}

		float cuda_running_configuration::get_device_saturation_time() const
		{
			return 5.0e-5F; // 50 us
		}

		cudnnConvolutionFwdAlgo_t cuda_running_configuration::cudnn_find_convolution_forward_algo(
			const cudnnTensorDescriptor_t input_desc,
			const cudnnFilterDescriptor_t weights_desc,
			const cudnnConvolutionDescriptor_t convolution_desc,
			const cudnnTensorDescriptor_t output_desc,
			const void * input_buffer,
			const void * weights,
			void * output_buffer,
			void * workspace,
			size_t workspace_size) const
		{
			key_convolution_param param;
			param.input_tensor_params = cudnn_util::get_tensor_params(input_desc);
			param.output_tensor_params = cudnn_util::get_tensor_params(output_desc);
			param.weights_params = cudnn_util::get_filter_params(weights_desc);
			param.conv_params = cudnn_util::get_convolution_params(convolution_desc);

			std::map<key_convolution_param, std::pair<bool, cudnnConvolutionFwdAlgo_t> >::const_iterator it = forward_param_to_best_algo_map.find(param);
			bool should_search = true;
			if (it != forward_param_to_best_algo_map.end())
			{
				if (it->second.first)
					return it->second.second;
				else
					should_search = false;
			}

			if (output_buffer && should_search)
			{
				int returned_algo_count;
				cudnnConvolutionFwdAlgoPerf_t perf_results;
				cuda_safe_call(cudaDeviceSynchronize()); // Make sure all previously submitted work doesn't interfere with finding the best algo
				cudnn_safe_call(cudnnFindConvolutionForwardAlgorithmEx(
					get_cudnn_handle(),
					input_desc,
					input_buffer,
					weights_desc,
					weights,
					convolution_desc,
					output_desc,
					output_buffer,
					1,
					&returned_algo_count,
					&perf_results,
					workspace,
					workspace_size));
				if (returned_algo_count == 1)
				{
					cudnnConvolutionFwdAlgo_t algo = perf_results.algo;
					forward_param_to_best_algo_map.insert(std::make_pair(param, std::make_pair(true, algo)));
					return algo;
				}
				else
				{
					forward_param_to_best_algo_map.insert(std::make_pair(param, std::make_pair(false, cudnnConvolutionFwdAlgo_t())));
				}
			}

			{
				cudnnConvolutionFwdAlgo_t algo;
				cudnn_safe_call(cudnnGetConvolutionForwardAlgorithm(
					get_cudnn_handle(),
					input_desc,
					weights_desc,
					convolution_desc,
					output_desc,
					workspace_size ? CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT : CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
					workspace_size,
					&algo));
				return algo;
			}
		}

		cudnnConvolutionBwdFilterAlgo_t cuda_running_configuration::cudnn_find_convolution_backward_weights_algo(
			const cudnnTensorDescriptor_t input_desc,
			const cudnnFilterDescriptor_t weights_desc,
			const cudnnConvolutionDescriptor_t convolution_desc,
			const cudnnTensorDescriptor_t output_desc,
			const void * input_buffer,
			const void * output_errors_buffer,
			void * gradients,
			void * workspace,
			size_t workspace_size) const
		{
			key_convolution_param param;
			param.input_tensor_params = cudnn_util::get_tensor_params(input_desc);
			param.output_tensor_params = cudnn_util::get_tensor_params(output_desc);
			param.weights_params = cudnn_util::get_filter_params(weights_desc);
			param.conv_params = cudnn_util::get_convolution_params(convolution_desc);

			std::map<key_convolution_param, std::pair<bool, cudnnConvolutionBwdFilterAlgo_t> >::const_iterator it = backward_weights_param_to_best_algo_map.find(param);
			bool should_search = true;
			if (it != backward_weights_param_to_best_algo_map.end())
			{
				if (it->second.first)
					return it->second.second;
				else
					should_search = false;
			}

			if (gradients && should_search)
			{
				int returned_algo_count;
				cudnnConvolutionBwdFilterAlgoPerf_t perf_results;
				cuda_safe_call(cudaDeviceSynchronize()); // Make sure all previously submitted work doesn't interfere with finding the best algo
				cudnn_safe_call(cudnnFindConvolutionBackwardFilterAlgorithmEx(
					get_cudnn_handle(),
					input_desc,
					input_buffer,
					output_desc,
					output_errors_buffer,
					convolution_desc,
					weights_desc,
					gradients,
					1,
					&returned_algo_count,
					&perf_results,
					workspace,
					workspace_size));
				if (returned_algo_count == 1)
				{
					cudnnConvolutionBwdFilterAlgo_t algo = perf_results.algo;
					backward_weights_param_to_best_algo_map.insert(std::make_pair(param, std::make_pair(true, algo)));
					return algo;
				}
				else
				{
					backward_weights_param_to_best_algo_map.insert(std::make_pair(param, std::make_pair(false, cudnnConvolutionBwdFilterAlgo_t())));
				}
			}

			{
				cudnnConvolutionBwdFilterAlgo_t algo;
				cudnn_safe_call(cudnnGetConvolutionBackwardFilterAlgorithm(
					get_cudnn_handle(),
					input_desc,
					output_desc,
					convolution_desc,
					weights_desc,
					workspace_size ? CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT : CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,
					workspace_size,
					&algo));
				return algo;
			}
		}

		cudnnConvolutionBwdDataAlgo_t cuda_running_configuration::cudnn_find_convolution_backward_data_algo(
			const cudnnTensorDescriptor_t input_desc,
			const cudnnFilterDescriptor_t weights_desc,
			const cudnnConvolutionDescriptor_t convolution_desc,
			const cudnnTensorDescriptor_t output_desc,
			const void * output_errors_buffer,
			const void * weights,
			void * input_errors_buffer,
			void * workspace,
			size_t workspace_size) const
		{
			key_convolution_param param;
			param.input_tensor_params = cudnn_util::get_tensor_params(input_desc);
			param.output_tensor_params = cudnn_util::get_tensor_params(output_desc);
			param.weights_params = cudnn_util::get_filter_params(weights_desc);
			param.conv_params = cudnn_util::get_convolution_params(convolution_desc);

			std::map<key_convolution_param, std::pair<bool, cudnnConvolutionBwdDataAlgo_t> >::const_iterator it = backward_data_param_to_best_algo_map.find(param);
			bool should_search = true;
			if (it != backward_data_param_to_best_algo_map.end())
			{
				if (it->second.first)
					return it->second.second;
				else
					should_search = false;
			}

			if (input_errors_buffer && should_search)
			{
				int returned_algo_count;
				cudnnConvolutionBwdDataAlgoPerf_t perf_results;
				cuda_safe_call(cudaDeviceSynchronize()); // Make sure all previously submitted work doesn't interfere with finding the best algo
				cudnn_safe_call(cudnnFindConvolutionBackwardDataAlgorithmEx(
					get_cudnn_handle(),
					weights_desc,
					weights,
					output_desc,
					output_errors_buffer,
					convolution_desc,
					input_desc,
					input_errors_buffer,
					1,
					&returned_algo_count,
					&perf_results,
					workspace,
					workspace_size));
				if (returned_algo_count == 1)
				{
					cudnnConvolutionBwdDataAlgo_t algo = perf_results.algo;
					backward_data_param_to_best_algo_map.insert(std::make_pair(param, std::make_pair(true, algo)));
					return algo;
				}
				else
				{
					backward_data_param_to_best_algo_map.insert(std::make_pair(param, std::make_pair(false, cudnnConvolutionBwdDataAlgo_t())));
				}
			}

			{
				cudnnConvolutionBwdDataAlgo_t algo;
				cudnn_safe_call(cudnnGetConvolutionBackwardDataAlgorithm(
					get_cudnn_handle(),
					weights_desc,
					output_desc,
					convolution_desc,
					input_desc,
					workspace_size ? CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT : CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,
					workspace_size,
					&algo));
				return algo;
			}
		}

		bool operator<(const key_convolution_param&x, const key_convolution_param&y)
		{
			if (x.input_tensor_params < y.input_tensor_params)
				return true;
			else if (y.input_tensor_params < x.input_tensor_params)
				return false;

			if (x.output_tensor_params < y.output_tensor_params)
				return true;
			else if (y.output_tensor_params < x.output_tensor_params)
				return false;

			if (x.weights_params < y.weights_params)
				return true;
			else if (y.weights_params < x.weights_params)
				return false;

			if (x.conv_params < y.conv_params)
				return true;
			else if (y.conv_params < x.conv_params)
				return false;

			return false;
		}

		void cuda_running_configuration::enqueue_reduce_all(
			const char * name,
			cuda_linear_buffer_device::ptr data,
			cuda_stream::ptr stream)
		{
			communicator->enqueue_reduce_all(
				name,
				device_pos,
				data,
				stream);
		}
	}
}
