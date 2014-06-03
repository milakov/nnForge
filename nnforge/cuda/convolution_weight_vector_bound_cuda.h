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

#include "weight_vector_bound_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		class convolution_weight_vector_bound_cuda: public weight_vector_bound_cuda
		{
		public:
			convolution_weight_vector_bound_cuda();

			virtual ~convolution_weight_vector_bound_cuda();

			virtual const boost::uuids::uuid& get_uuid() const;

			virtual void enqueue_normalize_weights(
				cudaStream_t stream_id,
				const weight_vector_bound& bound,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& data,
				const std::vector<cuda_linear_buffer_device_smart_ptr>& additional_buffers,
				unsigned int entry_count,
				const std::vector<unsigned int>& incoming_weight_count_per_output_neuron_list);

		protected:
			virtual weight_vector_bound_cuda_smart_ptr create_specific() const;

			// The method is called when configuration is finished
			virtual void weight_vector_bound_configured();

			int output_feature_map_count;

		private:
			static int get_threadblock_size(int output_neuron_count);
		};
	}
}
