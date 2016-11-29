/*
 *  Copyright 2011-2016 Maxim Milakov
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

#pragma once

#include <string>
#include <vector>

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cudnn.h>
#include <curand.h>
#include <memory>

#include "buffer_cuda_size_configuration.h"
#include "cuda_communicator.h"
#include "cudnn_util.h"

namespace nnforge
{
	namespace cuda
	{
		struct key_convolution_param
		{
			tensor_params input_tensor_params;
			tensor_params output_tensor_params;
			filter_params weights_params;
			convolution_params conv_params;
		};

		bool operator<(const key_convolution_param&x, const key_convolution_param&y);

		class cuda_running_configuration
		{
		public:
			typedef std::shared_ptr<cuda_running_configuration> ptr;
			typedef std::shared_ptr<const cuda_running_configuration> const_ptr;

			cuda_running_configuration(
				int device_id,
				float max_global_memory_usage_ratio,
				float cuda_fixed_working_buffers_ratio,
				int device_pos,
				cuda_communicator::ptr communicator);

			~cuda_running_configuration();

			size_t get_max_fixed_working_buffers_size() const;

			unsigned int get_max_entry_count(
				const buffer_cuda_size_configuration& buffers_config,
				float ratio = 1.0F) const;

			cublasHandle_t get_cublas_handle() const;

			cusparseHandle_t get_cusparse_handle() const;

			cudnnHandle_t get_cudnn_handle() const;

			curandGenerator_t get_curand_generator() const;

			bool is_flush_required() const;

			int get_compute_capability() const
			{
				return compute_capability_major * 100 + compute_capability_minor;
			}

			void set_device() const;

			float get_flops() const;

			float get_device_saturation_time() const;

			cudnnConvolutionFwdAlgo_t cudnn_find_convolution_forward_algo(
				const cudnnTensorDescriptor_t input_desc,
				const cudnnFilterDescriptor_t weights_desc,
				const cudnnConvolutionDescriptor_t convolution_desc,
				const cudnnTensorDescriptor_t output_desc,
				const void * input_buffer,
				const void * weights,
				void * output_buffer,
				void * workspace,
				size_t workspace_size) const;

			cudnnConvolutionBwdFilterAlgo_t cudnn_find_convolution_backward_weights_algo(
				const cudnnTensorDescriptor_t input_desc,
				const cudnnFilterDescriptor_t weights_desc,
				const cudnnConvolutionDescriptor_t convolution_desc,
				const cudnnTensorDescriptor_t output_desc,
				const void * input_buffer,
				const void * output_errors_buffer,
				void * gradients,
				void * workspace,
				size_t workspace_size) const;

			cudnnConvolutionBwdDataAlgo_t cudnn_find_convolution_backward_data_algo(
				const cudnnTensorDescriptor_t input_desc,
				const cudnnFilterDescriptor_t weights_desc,
				const cudnnConvolutionDescriptor_t convolution_desc,
				const cudnnTensorDescriptor_t output_desc,
				const void * output_errors_buffer,
				const void * weights,
				void * input_errors_buffer,
				void * workspace,
				size_t workspace_size) const;

			void enqueue_reduce_all(
				const char * name,
				cuda_linear_buffer_device::ptr data,
				cuda_stream::ptr stream);

		public:
			int device_id;
			float max_global_memory_usage_ratio;
			float cuda_fixed_working_buffers_ratio;
			int device_pos;
			cuda_communicator::ptr communicator;

			std::string device_name;
			int compute_capability_major;
			int compute_capability_minor;
			int clock_rate; // in kHz
			int memory_clock_rate; // in kHz
			int memory_bus_width; // in bits
			size_t global_memory_size;
			bool ecc_enabled;
			int l2_cache_size;
			int multiprocessor_count;
			size_t smem_per_block;
			int max_threads_per_multiprocessor;
			int max_threads_per_block;
			int max_threads_dim[3];
			int max_grid_size[3];
			int max_texture_1d_linear;
			int texture_alignment; // in bytes
			int pci_bus_id;
			int pci_device_id;
			int least_stream_priority;
			int greatest_stream_priority;

		#ifdef _WIN32
			bool tcc_mode;
		#endif

		private:
			int get_core_count_per_sm() const;

		private:
			cuda_running_configuration() = delete;
			cuda_running_configuration(const cuda_running_configuration&) = delete;
			cuda_running_configuration& operator =(const cuda_running_configuration&) = delete;

			void update_parameters();

			cublasHandle_t cublas_handle;
			cusparseHandle_t cusparse_handle;
			cudnnHandle_t cudnn_handle;
			curandGenerator_t curand_gen;

			mutable std::map<key_convolution_param, std::pair<bool, cudnnConvolutionFwdAlgo_t> > forward_param_to_best_algo_map;
			mutable std::map<key_convolution_param, std::pair<bool, cudnnConvolutionBwdFilterAlgo_t> > backward_weights_param_to_best_algo_map;
			mutable std::map<key_convolution_param, std::pair<bool, cudnnConvolutionBwdDataAlgo_t> > backward_data_param_to_best_algo_map;
		};

		std::ostream& operator<< (std::ostream& out, const cuda_running_configuration& running_configuration);
	}
}
