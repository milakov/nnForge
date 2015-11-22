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

#pragma once

#include <string>

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cudnn.h>
#include <curand.h>

#include "buffer_cuda_size_configuration.h"
#include "../nn_types.h"
#include "../threadpool_job_runner.h"

namespace nnforge
{
	namespace cuda
	{
		class cuda_running_configuration
		{
		public:
			typedef nnforge_shared_ptr<cuda_running_configuration> ptr;
			typedef nnforge_shared_ptr<const cuda_running_configuration> const_ptr;

			cuda_running_configuration(
				int device_id,
				float max_global_memory_usage_ratio,
				unsigned int reserved_thread_count,
				bool dont_share_buffers,
				bool single_command_stream,
				unsigned int optimize_action_graph_assumed_chunk_size);

			~cuda_running_configuration();

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

			threadpool_job_runner::ptr get_job_runner() const;

			bool is_dont_share_buffers() const;

			bool is_single_command_stream() const;

			float get_flops() const;

			float get_device_saturation_time() const;

		public:
			int device_id;
			float max_global_memory_usage_ratio;
			unsigned int reserved_thread_count;
			bool dont_share_buffers;
			bool single_command_stream;
			unsigned int optimize_action_graph_assumed_chunk_size;

			int driver_version;
			int runtime_version;

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

		#ifdef _WIN32
			bool tcc_mode;
		#endif

		private:
			int get_core_count_per_sm() const;

		private:
			cuda_running_configuration();
			cuda_running_configuration(const cuda_running_configuration&);
			cuda_running_configuration& operator =(const cuda_running_configuration&);

			void update_parameters();

			cublasHandle_t cublas_handle;
			cusparseHandle_t cusparse_handle;
			cudnnHandle_t cudnn_handle;
			curandGenerator_t curand_gen;

			threadpool_job_runner::ptr job_runner;
		};

		std::ostream& operator<< (std::ostream& out, const cuda_running_configuration& running_configuration);
	}
}
