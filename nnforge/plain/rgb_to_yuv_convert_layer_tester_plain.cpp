/*
 *  Copyright 2011-2015 Maxim Milakov
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

#include "rgb_to_yuv_convert_layer_tester_plain.h"

#include "../rgb_to_yuv_convert_layer.h"
#include "../nn_types.h"

#include <cstring>

#define w_r 0.299F
#define w_b 0.114F
#define w_g (1.0F - w_r - w_b)
#define u_max 0.436F
#define v_max 0.615F
#define u_mult (u_max / (1.0F - w_b))
#define v_mult (v_max / (1.0F - w_r))

namespace nnforge
{
	namespace plain
	{
		rgb_to_yuv_convert_layer_tester_plain::rgb_to_yuv_convert_layer_tester_plain()
		{
		}

		rgb_to_yuv_convert_layer_tester_plain::~rgb_to_yuv_convert_layer_tester_plain()
		{
		}

		std::string rgb_to_yuv_convert_layer_tester_plain::get_type_name() const
		{
			return rgb_to_yuv_convert_layer::layer_type_name;
		}

		void rgb_to_yuv_convert_layer_tester_plain::run_forward_propagation(
			plain_buffer::ptr output_buffer,
			const std::vector<plain_buffer::const_ptr>& input_buffers,
			plain_buffer::ptr temporary_working_fixed_buffer,
			plain_buffer::ptr temporary_working_per_entry_buffer,
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			layer_data::const_ptr data,
			layer_data_custom::const_ptr data_custom,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int entry_count) const
		{
			const float * const in_it = *input_buffers[0];
			float * const out_it = *output_buffer;

			nnforge_shared_ptr<const rgb_to_yuv_convert_layer> layer_derived = nnforge_dynamic_pointer_cast<const rgb_to_yuv_convert_layer>(layer_schema);

			const unsigned int color_feature_map_config_count = static_cast<unsigned int>(layer_derived->color_feature_map_config_list.size());

			if ((out_it != in_it) && ((color_feature_map_config_count * 3) != output_configuration_specific.feature_map_count))
				memcpy(out_it, in_it, output_configuration_specific.get_neuron_count() * entry_count * sizeof(float));

			const int total_workload = static_cast<int>(entry_count * color_feature_map_config_count);

			const unsigned int input_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			const std::vector<color_feature_map_config>::const_iterator cfm_it = layer_derived->color_feature_map_config_list.begin();

			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int workload_id = 0; workload_id < total_workload; ++workload_id)
			{
				int entry_id = workload_id / color_feature_map_config_count;
				int color_feature_map_config_id = workload_id - entry_id * color_feature_map_config_count;
				const color_feature_map_config& cfm = *(cfm_it + color_feature_map_config_id);

				const float * in_it_red_and_y = in_it + (entry_id * input_neuron_count) + (cfm.red_and_y_feature_map_id * input_neuron_count_per_feature_map);
				const float * in_it_green_and_u = in_it + (entry_id * input_neuron_count) + (cfm.green_and_u_feature_map_id * input_neuron_count_per_feature_map);
				const float * in_it_blue_and_v = in_it + (entry_id * input_neuron_count) + (cfm.blue_and_v_feature_map_id * input_neuron_count_per_feature_map);

				float * out_it_red_and_y = out_it + (entry_id * input_neuron_count) + (cfm.red_and_y_feature_map_id * input_neuron_count_per_feature_map);
				float * out_it_green_and_u = out_it + (entry_id * input_neuron_count) + (cfm.green_and_u_feature_map_id * input_neuron_count_per_feature_map);
				float * out_it_blue_and_v = out_it + (entry_id * input_neuron_count) + (cfm.blue_and_v_feature_map_id * input_neuron_count_per_feature_map);

				for(unsigned int i = 0; i < input_neuron_count_per_feature_map; ++i)
				{
					float red = in_it_red_and_y[i];
					float green = in_it_green_and_u[i];
					float blue = in_it_blue_and_v[i];

					float y = w_r * red + w_g * green + w_b * blue;
					float u = u_mult * (blue - y);
					float v = v_mult * (red - y);

					out_it_red_and_y[i] = y;
					out_it_green_and_u[i] = u;
					out_it_blue_and_v[i] = v;
				}
			}
		}

		int rgb_to_yuv_convert_layer_tester_plain::get_input_index_layer_can_write(
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return 0;
		}
	}
}
