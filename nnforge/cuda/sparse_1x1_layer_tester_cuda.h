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

#include "layer_tester_cuda.h"

#include <cudnn.h>

namespace nnforge
{
	namespace cuda
	{
		class sparse_1x1_layer_tester_cuda : public layer_tester_cuda
		{
		public:
			sparse_1x1_layer_tester_cuda();

			virtual ~sparse_1x1_layer_tester_cuda();

			virtual void enqueue_forward_propagation(
				cudaStream_t stream_id,
				cuda_linear_buffer_device::ptr output_buffer,
				const std::vector<cuda_linear_buffer_device::const_ptr>& schema_data,
				const std::vector<cuda_linear_buffer_device::const_ptr>& data,
				const std::vector<cuda_linear_buffer_device::const_ptr>& data_custom,
				const std::vector<cuda_linear_buffer_device::const_ptr>& input_buffers,
				const std::vector<cuda_linear_buffer_device::const_ptr>& persistent_working_data,
				cuda_linear_buffer_device::ptr temporary_working_fixed_buffer,
				cuda_linear_buffer_device::ptr temporary_working_per_entry_buffer,
				unsigned int entry_count);

			virtual size_t get_temporary_working_per_entry_buffer_size() const;

		protected:
			virtual void tester_configured();

		private:
			int feature_map_connection_count;
			bool bias;
			bool unit_stride;
			layer_configuration_specific input_strided_config;
			std::vector<unsigned int> input_strides;
			std::vector<unsigned int> input_converted_NHWC_strides;
			std::vector<unsigned int> input_converted_CNHW_strides_base;

			cudnnTensorDescriptor_t input_strided_data_desc;
			cudnnTensorDescriptor_t input_converted_NHWC_data_desc;
			cudnnTensorDescriptor_t input_converted_CNHW_data_desc;
			cudnnTensorDescriptor_t output_data_desc;
			cudnnTensorDescriptor_t bias_desc;

			int input_converted_elem_count_per_entry_aligned;
			int output_elem_count_per_entry_aligned;
		};
	}
}
