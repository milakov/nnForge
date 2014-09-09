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

#include "sparse_fully_connected_1x1_layer_tester_cuda.h"

#include <cuda_runtime.h>

#include "util_cuda.h"
#include "neural_network_cusparse_exception.h"
#include "../sparse_convolution_layer.h"

namespace nnforge
{
	namespace cuda
	{
		sparse_fully_connected_1x1_layer_tester_cuda::sparse_fully_connected_1x1_layer_tester_cuda()
		{
		}

		sparse_fully_connected_1x1_layer_tester_cuda::~sparse_fully_connected_1x1_layer_tester_cuda()
		{
		}

		void sparse_fully_connected_1x1_layer_tester_cuda::enqueue_test(
			cudaStream_t stream_id,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& schema_data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data,
			const std::vector<const_cuda_linear_buffer_device_smart_ptr>& data_custom,
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
			unsigned int entry_count)
		{
			// Copy bias
			cuda_util::duplicate_vector(
				*cuda_config,
				*data[1],
				*additional_buffers[0],
				output_elem_count_per_entry,
				entry_count,
				stream_id);

			cusparse_safe_call(cusparseSetStream(cuda_config->get_cusparse_handle(), stream_id));
			float alpha = 1.0F;
			float beta = 1.0F;
			cusparseMatDescr_t mat_descr;
			cusparse_safe_call(cusparseCreateMatDescr(&mat_descr));
			cusparse_safe_call(cusparseScsrmm(
				cuda_config->get_cusparse_handle(),
				CUSPARSE_OPERATION_NON_TRANSPOSE,
				output_elem_count_per_entry,
				entry_count,
				input_elem_count_per_entry,
				feature_map_connection_count,
				&alpha,
				mat_descr,
				*data[0],
				*data_custom[1],
				*data_custom[0],
				*input_buffer,
				input_elem_count_per_entry,
				&beta,
				*additional_buffers[0],
				output_elem_count_per_entry));
		}

		std::vector<size_t> sparse_fully_connected_1x1_layer_tester_cuda::get_sizes_of_additional_buffers_per_entry() const
		{
			std::vector<size_t> res;

			res.push_back(output_elem_count_per_entry * sizeof(float));

			return res;
		}

		cuda_linear_buffer_device_smart_ptr sparse_fully_connected_1x1_layer_tester_cuda::get_output_buffer(
			cuda_linear_buffer_device_smart_ptr input_buffer,
			const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers)
		{
			return additional_buffers[0];
		}

		void sparse_fully_connected_1x1_layer_tester_cuda::tester_configured()
		{
			nnforge_shared_ptr<const sparse_convolution_layer> layer_derived = nnforge_dynamic_pointer_cast<const sparse_convolution_layer>(layer_schema);

			feature_map_connection_count = layer_derived->feature_map_connection_count;
		}
	}
}
