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

#include "maxout_layer_updater_plain.h"

#include "../maxout_layer.h"
#include "../neural_network_exception.h"
#include "../nn_types.h"

#include <array>

namespace nnforge
{
	namespace plain
	{
		maxout_layer_updater_plain::maxout_layer_updater_plain()
		{
		}

		maxout_layer_updater_plain::~maxout_layer_updater_plain()
		{
		}

		const boost::uuids::uuid& maxout_layer_updater_plain::get_uuid() const
		{
			return maxout_layer::layer_guid;
		}

		void maxout_layer_updater_plain::test(
			const_additional_buffer_smart_ptr input_buffer,
			additional_buffer_smart_ptr output_buffer,
			std::vector<additional_buffer_smart_ptr>& additional_buffers,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int updater_count,
			unsigned int offset_input_entry_id) const
		{
			if (offset_input_entry_id > 0)
				throw neural_network_exception("maxout_layer_updater_plain is not able to run using offset");

			const std::vector<float>::const_iterator in_it_global = input_buffer->begin();
			const std::vector<float>::iterator out_it_global = output_buffer->begin();
			const std::vector<float>::iterator max_feature_map_positions_it_global = additional_buffers[0]->begin();

			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			nnforge_shared_ptr<const maxout_layer> layer_derived = nnforge_dynamic_pointer_cast<const maxout_layer>(layer_schema);
			const unsigned int feature_map_subsampling_size = layer_derived->feature_map_subsampling_size;
			const int output_feature_map_count = output_configuration_specific.feature_map_count;
			const int total_workload = updater_count * output_feature_map_count;

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / output_feature_map_count;
					int output_feature_map_id = workload_id - (entry_id * output_feature_map_count);

					std::vector<float>::const_iterator in_it_base = in_it_global + (entry_id * input_neuron_count) + (output_feature_map_id * input_neuron_count_per_feature_map);
					int output_offset = (entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);
					std::vector<float>::iterator out_it_base = out_it_global + output_offset;
					std::vector<float>::iterator max_feature_map_positions_it = max_feature_map_positions_it_global + output_offset;

					for(std::vector<float>::iterator out_it = out_it_base; out_it != out_it_base + output_neuron_count_per_feature_map; ++out_it, ++max_feature_map_positions_it, ++in_it_base)
					{
						std::vector<float>::const_iterator in_it = in_it_base;
						float current_max = *in_it;
						int max_feature_map_pos = 0;
						for(unsigned int i = 1; i < feature_map_subsampling_size; ++i)
						{
							in_it += output_feature_map_count * output_neuron_count_per_feature_map;
							float new_val = *in_it;
							if (new_val > current_max)
							{
								current_max = new_val;
								max_feature_map_pos = i;
							}
						}
						*out_it = current_max;
						*((unsigned int *)(&(*max_feature_map_positions_it))) = max_feature_map_pos;
					}
				}
			}
		}

		void maxout_layer_updater_plain::backprop(
			additional_buffer_smart_ptr input_errors,
			const_additional_buffer_smart_ptr input_neurons,
			const_additional_buffer_smart_ptr output_errors,
			const_additional_buffer_smart_ptr output_neurons,
			std::vector<additional_buffer_smart_ptr>& additional_buffers,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int updater_count) const
		{
			const std::vector<float>::iterator in_err_it_global = input_errors->begin();
			const std::vector<float>::const_iterator out_err_it_global = output_errors->begin();
			const std::vector<float>::const_iterator max_feature_map_positions_it_global = additional_buffers[0]->begin();

			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			nnforge_shared_ptr<const maxout_layer> layer_derived = nnforge_dynamic_pointer_cast<const maxout_layer>(layer_schema);
			const unsigned int feature_map_subsampling_size = layer_derived->feature_map_subsampling_size;
			const int output_feature_map_count = output_configuration_specific.feature_map_count;
			const int total_workload = updater_count * output_feature_map_count;

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / output_feature_map_count;
					int output_feature_map_id = workload_id - (entry_id * output_feature_map_count);

					std::vector<float>::iterator in_err_it_base = in_err_it_global + (entry_id * input_neuron_count) + (output_feature_map_id * input_neuron_count_per_feature_map);
					int output_offset = (entry_id * output_neuron_count) + (output_feature_map_id * output_neuron_count_per_feature_map);
					std::vector<float>::const_iterator out_err_it_base = out_err_it_global + output_offset;
					std::vector<float>::const_iterator max_feature_map_positions_it = max_feature_map_positions_it_global + output_offset;

					for(std::vector<float>::const_iterator out_err_it = out_err_it_base; out_err_it != out_err_it_base + output_neuron_count_per_feature_map; ++out_err_it, ++max_feature_map_positions_it, ++in_err_it_base)
					{
						std::vector<float>::iterator in_err_it = in_err_it_base;
						float current_err = *out_err_it;
						unsigned int max_feature_map_position = *((const unsigned int *)(&(*max_feature_map_positions_it)));
						for(unsigned int i = 0; i < feature_map_subsampling_size; ++i)
							in_err_it[output_feature_map_count * output_neuron_count_per_feature_map * i] = ((i == max_feature_map_position) ? current_err : 0.0F);
					}
				}
			}
		}

		std::vector<std::pair<unsigned int, bool> > maxout_layer_updater_plain::get_elem_count_and_per_entry_flag_additional_buffers(
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			plain_running_configuration_const_smart_ptr plain_config,
			bool backprop_required) const
		{
			std::vector<std::pair<unsigned int, bool> > res;

			if (backprop_required)
				res.push_back(std::make_pair<unsigned int, bool>(output_configuration_specific.get_neuron_count(), true));

			return res;
		}

		bool maxout_layer_updater_plain::is_in_place_backprop() const
		{
			return false;
		}
	}
}
