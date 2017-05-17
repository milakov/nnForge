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

#pragma once

#include "layer_tester_cuda.h"

#include <cudnn.h>

namespace nnforge
{
	namespace cuda
	{
		class convolution_layer_tester_cuda : public layer_tester_cuda
		{
		public:
			convolution_layer_tester_cuda();

			virtual ~convolution_layer_tester_cuda();

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

			virtual std::pair<size_t, bool> get_temporary_working_fixed_buffer_size() const;

		protected:
			virtual void tester_configured();

			std::vector<unsigned int> window_sizes;
			std::vector<unsigned int> strides;
			std::vector<unsigned int> dilation;
			bool bias;

			cudnnTensorDescriptor_t input_data_desc;
			cudnnTensorDescriptor_t output_data_desc;
			cudnnFilterDescriptor_t weights_desc;
			cudnnConvolutionDescriptor_t convolution_desc;
			cudnnTensorDescriptor_t bias_desc;
		};
	}
}
