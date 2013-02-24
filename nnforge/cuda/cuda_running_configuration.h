/*
 *  Copyright 2011-2013 Maxim Milakov
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

#include <memory>
#include <string>

#include <cublas_v2.h>

#include "buffer_cuda_size_configuration.h"

namespace nnforge
{
	namespace cuda
	{
		class cuda_running_configuration
		{
		public:
			cuda_running_configuration(float max_global_memory_usage_ratio);

			~cuda_running_configuration();

			unsigned int get_max_entry_count(
				const buffer_cuda_size_configuration& buffers_config,
				float ratio = 1.0F) const;

			cublasHandle_t get_cublas_handle() const;

			bool is_flush_required() const;

		public:
			float max_global_memory_usage_ratio;

			int driver_version;
			int runtime_version;

			int device_id;
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
			int pci_bus_id;
			int pci_device_id;

		#ifdef WIN32
			bool tcc_mode;
		#endif

		private:
			cuda_running_configuration();
			cuda_running_configuration(const cuda_running_configuration&);
			cuda_running_configuration& operator =(const cuda_running_configuration&);

			void update_parameters();

			cublasHandle_t cublas_handle;
		};

		typedef std::tr1::shared_ptr<const cuda_running_configuration> cuda_running_configuration_const_smart_ptr;

		std::ostream& operator<< (std::ostream& out, const cuda_running_configuration& running_configuration);
	}
}
