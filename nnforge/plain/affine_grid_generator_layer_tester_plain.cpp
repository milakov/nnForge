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

#include "affine_grid_generator_layer_tester_plain.h"

#include "../affine_grid_generator_layer.h"
#include "../nn_types.h"

#include <array>

namespace nnforge
{
	namespace plain
	{
		affine_grid_generator_layer_tester_plain::affine_grid_generator_layer_tester_plain()
		{
		}

		affine_grid_generator_layer_tester_plain::~affine_grid_generator_layer_tester_plain()
		{
		}

		std::string affine_grid_generator_layer_tester_plain::get_type_name() const
		{
			return affine_grid_generator_layer::layer_type_name;
		}

		void affine_grid_generator_layer_tester_plain::run_forward_propagation(
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
			const float * const in_it_global = *input_buffers[0];
			float * const out_it_global = *output_buffer;
			const unsigned int input_neuron_count = input_configuration_specific_list[0].get_neuron_count();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			nnforge_shared_ptr<const affine_grid_generator_layer> layer_derived = nnforge_dynamic_pointer_cast<const affine_grid_generator_layer>(layer_schema);
			const unsigned int output_width = layer_derived->output_sizes[0];
			const unsigned int output_height = layer_derived->output_sizes[1];
			const bool adjust_for_zero_init = layer_derived->adjust_for_zero_init;
			const float x_scale = output_width > 1 ? 1.0F / static_cast<float>(output_width - 1) : 1.0F;
			const float y_scale = output_height > 1 ? 1.0F / static_cast<float>(output_height - 1) : 1.0F;
			const float weight_scale = layer_derived->get_weight_scale(output_configuration_specific);
			const int total_workload = entry_count * output_height;

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / output_height;
					int y = workload_id - (entry_id * output_height);
					float y_pos = (float)y * y_scale;

					const float * in_it_base = in_it_global + (entry_id * input_neuron_count);
					float input_vals[6];
					for(int i = 0; i < 6; ++i)
						input_vals[i] = in_it_base[i] * weight_scale;
					if (adjust_for_zero_init)
					{
						input_vals[0] += 1.0F;
						input_vals[4] += 1.0F;
					}

					float * out_it_base_x = out_it_global + entry_id * output_neuron_count + y * output_width;
					float * out_it_base_y = out_it_base_x + output_neuron_count_per_feature_map;

					float x_out_pos_base = y_pos * input_vals[1] + input_vals[2];
					float y_out_pos_base = y_pos * input_vals[4] + input_vals[5];

					for(unsigned int x = 0; x < output_width; ++x)
					{
						float x_pos = (float)x * x_scale;

						float x_out_pos = x_pos * input_vals[0] + x_out_pos_base;
						float y_out_pos = x_pos * input_vals[3] + y_out_pos_base;

						out_it_base_x[x] = x_out_pos;
						out_it_base_y[x] = y_out_pos;
					}
				}
			}
		}
	}
}
