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

#include "convolution_weight_vector_bound_plain.h"

#include "../convolution_layer.h"

#include <numeric>

namespace nnforge
{
	namespace plain
	{
		convolution_weight_vector_bound_plain::convolution_weight_vector_bound_plain()
		{
		}

		convolution_weight_vector_bound_plain::~convolution_weight_vector_bound_plain()
		{
		}

		const boost::uuids::uuid& convolution_weight_vector_bound_plain::get_uuid() const
		{
			return convolution_layer::layer_guid;
		}

		void convolution_weight_vector_bound_plain::normalize_weights(
			const weight_vector_bound& bound,
			layer_data_list& data,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			unsigned int updater_count) const
		{
			std::tr1::shared_ptr<const convolution_layer> layer_derived = std::tr1::dynamic_pointer_cast<const convolution_layer>(layer_schema);

			unsigned int weight_count = 1;
			for(std::vector<unsigned int>::const_iterator it = layer_derived->window_sizes.begin(); it != layer_derived->window_sizes.end(); ++it)
				weight_count *= *it;
			const unsigned int weight_block_size = weight_count * layer_derived->input_feature_map_count;
			const unsigned int output_feature_map_count = layer_derived->output_feature_map_count;
			const int total_workload = output_feature_map_count * updater_count;
			const layer_data_list::iterator data_list_it = data.begin();
			const float max_l2_norm_squared = bound.max_l2_norm * bound.max_l2_norm;
			const accum_helper_struct accum_helper;

			#pragma omp parallel for default(none) num_threads(plain_config->openmp_thread_count) schedule(guided)
			for(int workload_id = 0; workload_id < total_workload; ++workload_id)
			{
				int entry_id = workload_id / output_feature_map_count;
				int output_feature_map_id = workload_id - (entry_id * output_feature_map_count);

				std::vector<float>::iterator weights = (**(data_list_it + entry_id))[0].begin() + (output_feature_map_id * weight_block_size);

				float l2_norm_squared = std::accumulate(weights, weights + weight_block_size, 0.0F, accum_helper);
				if (l2_norm_squared > max_l2_norm_squared)
				{
					float mult = sqrtf(max_l2_norm_squared / l2_norm_squared);
					std::transform(weights, weights + weight_block_size, weights, scale_helper_struct(mult));
				}
			}
		}
	}
}
