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

#include "untile_layer_tester_plain.h"

#include "../untile_layer.h"
#include "../nn_types.h"
#include "../neural_network_exception.h"

#include <array>
#include <boost/format.hpp>

namespace nnforge
{
	namespace plain
	{
		untile_layer_tester_plain::untile_layer_tester_plain()
		{
		}

		untile_layer_tester_plain::~untile_layer_tester_plain()
		{
		}

		const boost::uuids::uuid& untile_layer_tester_plain::get_uuid() const
		{
			return untile_layer::layer_guid;
		}

		void untile_layer_tester_plain::test(
			additional_buffer_smart_ptr input_buffer,
			additional_buffer_set& additional_buffers,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const_layer_data_custom_smart_ptr data_custom,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int entry_count) const
		{
			const std::vector<float>::const_iterator in_it_global = input_buffer->begin();
			const std::vector<float>::iterator out_it_global = additional_buffers[0]->begin();
			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			nnforge_shared_ptr<const untile_layer> layer_derived = nnforge_dynamic_pointer_cast<const untile_layer>(layer_schema);
			const std::vector<std::vector<unsigned int> >& upsampling_sizes_list = layer_derived->upsampling_sizes_list;
			const int total_tiling_factor = layer_derived->get_tiling_factor().get_inverse();

			if (entry_count % total_tiling_factor != 0)
				throw neural_network_exception((boost::format("untile_layer_tester_plain: entry_count (%1%) is not evenly divisible by total_tiling_factor (%2%)") % entry_count % total_tiling_factor).str());

			std::vector<int> position_list(input_neuron_count_per_feature_map);
			{
				std::vector<unsigned int> tiling_sizes(input_configuration_specific.dimension_sizes.size(), 1);
				for(int i = 0; i < upsampling_sizes_list.size(); ++i)
				{
					const std::vector<unsigned int>& upsampling_sizes = upsampling_sizes_list[i];
					for(int j = 0; j < upsampling_sizes.size(); ++j)
						tiling_sizes[j] *= upsampling_sizes[j];
				}

				std::vector<unsigned int> spatial_pos(input_configuration_specific.dimension_sizes.size(), 0);
				for(unsigned int i = 0; i < input_neuron_count_per_feature_map; ++i)
				{
					unsigned int pos = spatial_pos.back() * tiling_sizes.back();
					for(int j = static_cast<int>(spatial_pos.size() - 2); j >= 0; --j)
						pos = pos * output_configuration_specific.dimension_sizes[j] + spatial_pos[j] * tiling_sizes[j];
					position_list[i] = pos;

					for(int j = 0; j < spatial_pos.size(); ++j)
					{
						if ((++spatial_pos[j]) < input_configuration_specific.dimension_sizes[j])
							break;
						spatial_pos[j] = 0;
					}
				}
			} // position_list

			std::vector<int> offset_list(total_tiling_factor);
			{
				std::vector<std::vector<unsigned int> > positions_list;
				positions_list.push_back(std::vector<unsigned int>(output_configuration_specific.dimension_sizes.size(), 0));

				std::vector<unsigned int> total_upsampling_sizes(upsampling_sizes_list.front().size(), 1);

				for(int level = static_cast<unsigned int>(upsampling_sizes_list.size()) - 1; level >= 0; --level)
				{
					std::vector<std::vector<unsigned int> > new_positions_list;
					const std::vector<unsigned int>& upsampling_sizes = upsampling_sizes_list[level];

					unsigned int local_tiling_count = 1;
					for(std::vector<unsigned int>::const_iterator it = upsampling_sizes.begin(); it != upsampling_sizes.end(); ++it)
						local_tiling_count *= *it;

					for(std::vector<std::vector<unsigned int> >::const_iterator it = positions_list.begin(); it != positions_list.end(); ++it)
					{
						const std::vector<unsigned int>& current_positions = *it;

						std::vector<unsigned int> local_pos(upsampling_sizes.size(), 0);
						for(unsigned int i = 0; i < local_tiling_count; ++i)
						{
							std::vector<unsigned int> new_untiled_positions(current_positions);
							for(unsigned int i = 0; i < static_cast<unsigned int>(upsampling_sizes.size()); ++i)
								new_untiled_positions[i] += local_pos[i] * total_upsampling_sizes[i];

							new_positions_list.push_back(new_untiled_positions);

							for(int j = 0; j < local_pos.size(); ++j)
							{
								if ((++local_pos[j]) < upsampling_sizes[j])
									break;
								local_pos[j] = 0;
							}
						}
					}

					for(unsigned int i = 0; i < static_cast<unsigned int>(total_upsampling_sizes.size()); ++i)
						total_upsampling_sizes[i] *= upsampling_sizes[i];

					positions_list = new_positions_list;
				}

				for(int i = 0; i < total_tiling_factor; ++i)
				{
					const std::vector<unsigned int>& positions = positions_list[i];
					int pos = positions.back();
					for(int j = static_cast<int>(positions.size() - 2); j >= 0; --j)
						pos = pos * output_configuration_specific.dimension_sizes[j] + positions[j];
					offset_list[i] = pos;
				}
			} // offset_list

			const int feature_map_count = output_configuration_specific.feature_map_count;
			const int output_entry_count = entry_count / total_tiling_factor;
			const int total_workload = output_entry_count * feature_map_count;
			const std::vector<int>::const_iterator position_list_it = position_list.begin();
			const std::vector<int>::const_iterator offset_list_it = offset_list.begin();

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					unsigned int output_entry_id = workload_id / feature_map_count;
					int feature_map_id = workload_id - (output_entry_id * feature_map_count);
					int base_input_entry_id = output_entry_id * total_tiling_factor;
					const std::vector<float>::iterator base_output_it = out_it_global + (output_entry_id * feature_map_count + feature_map_id) * output_neuron_count_per_feature_map;
					const std::vector<float>::const_iterator base_input_it = in_it_global + (base_input_entry_id * feature_map_count + feature_map_id) * input_neuron_count_per_feature_map;

					for(int input_neuron_id = 0; input_neuron_id < static_cast<int>(input_neuron_count_per_feature_map); ++input_neuron_id)
					{
						const std::vector<float>::iterator base_output_it2 = base_output_it + position_list_it[input_neuron_id];
						const std::vector<float>::const_iterator base_input_it2 = base_input_it + input_neuron_id;
						for(int local_entry_id = 0; local_entry_id < total_tiling_factor; ++local_entry_id)
						{
							float val = base_input_it2[local_entry_id * input_neuron_count];
							base_output_it2[offset_list_it[local_entry_id]] = val;
						}
					}
				}
			}
		}

		additional_buffer_smart_ptr untile_layer_tester_plain::get_output_buffer(
			additional_buffer_smart_ptr input_buffer,
			additional_buffer_set& additional_buffers) const
		{
			return additional_buffers[0];
		}

		std::vector<std::pair<unsigned int, bool> > untile_layer_tester_plain::get_elem_count_and_per_entry_flag_additional_buffers(
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			plain_running_configuration_const_smart_ptr plain_config) const
		{
			std::vector<std::pair<unsigned int, bool> > res;

			nnforge_shared_ptr<const untile_layer> layer_derived = nnforge_dynamic_pointer_cast<const untile_layer>(layer_schema);

			unsigned int total_tiling_factor = layer_derived->get_tiling_factor().get_inverse();
			res.push_back(std::make_pair<unsigned int, bool>((output_configuration_specific.get_neuron_count() + total_tiling_factor - 1) / total_tiling_factor, true));

			return res;
		}
	}
}
