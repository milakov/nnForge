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

#include "local_contrast_subtractive_layer_updater_plain.h"

#ifdef _OPENMP
#include <omp.h>
#endif

#include "../local_contrast_subtractive_layer.h"
#include "../neural_network_exception.h"
#include "../nn_types.h"

namespace nnforge
{
	namespace plain
	{
		local_contrast_subtractive_layer_updater_plain::local_contrast_subtractive_layer_updater_plain()
		{
		}

		local_contrast_subtractive_layer_updater_plain::~local_contrast_subtractive_layer_updater_plain()
		{
		}

		const boost::uuids::uuid& local_contrast_subtractive_layer_updater_plain::get_uuid() const
		{
			return local_contrast_subtractive_layer::layer_guid;
		}

		void local_contrast_subtractive_layer_updater_plain::test(
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
				throw neural_network_exception("local_contrast_subtractive_layer_updater_plain is not able to run using offset");

			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			nnforge_shared_ptr<const local_contrast_subtractive_layer> layer_derived = nnforge_dynamic_pointer_cast<const local_contrast_subtractive_layer>(layer_schema);
			const std::vector<std::vector<float> >& window_weights_list = layer_derived->window_weights_list;
			const std::vector<unsigned int>& feature_maps_affected = layer_derived->feature_maps_affected;
			const std::vector<unsigned int>& feature_maps_unaffected = layer_derived->feature_maps_unaffected;
			const unsigned int dimension_count = static_cast<unsigned int>(window_weights_list.size());
			std::vector<unsigned int> input_slices(input_configuration_specific.dimension_sizes.size());
			input_slices[0] = 1;
			for(unsigned int i = 0; i < dimension_count - 1; ++i)
				input_slices[i + 1] = input_slices[i] * input_configuration_specific.dimension_sizes[i];

			const std::vector<unsigned int>::const_iterator dimension_sizes_it = output_configuration_specific.dimension_sizes.begin();
			const unsigned int feature_maps_affected_count = static_cast<unsigned int>(feature_maps_affected.size());
			const unsigned int feature_maps_unaffected_count = static_cast<unsigned int>(feature_maps_affected.size());
			const std::vector<unsigned int>::const_iterator input_slices_it = input_slices.begin();
			const std::vector<unsigned int>::const_iterator feature_maps_affected_it = feature_maps_affected.begin();
			const std::vector<float>::const_iterator input_buffer_it = input_buffer->begin();
			const std::vector<float>::iterator output_buffer_it = output_buffer->begin();
			const std::vector<std::vector<float> >::const_iterator window_weights_list_it = window_weights_list.begin();

			const int total_workload = updater_count * feature_maps_affected_count;
			const int openmp_thread_count = plain_config->openmp_thread_count;
			
			#pragma omp parallel default(none) shared(additional_buffers) num_threads(openmp_thread_count)
			{
				std::vector<additional_buffer_smart_ptr> local_additional_buffers;
				int thread_id = 0;
				#ifdef _OPENMP
				thread_id = omp_get_thread_num();
				#endif

				local_additional_buffers.push_back(additional_buffers[thread_id]);
				if (dimension_count > 1)
					local_additional_buffers.push_back(additional_buffers[openmp_thread_count + thread_id]);

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / feature_maps_affected_count;
					int affected_feature_map_id = workload_id - (entry_id * feature_maps_affected_count);

					unsigned int current_output_buffer_index = 0;
					unsigned int feature_map_id = *(feature_maps_affected_it + affected_feature_map_id);
					for(unsigned int dimension_id = 0; dimension_id < dimension_count; ++dimension_id)
					{
						std::vector<float>::iterator out_it_base = local_additional_buffers[current_output_buffer_index]->begin();
						std::vector<float>::const_iterator in_it;
						if (dimension_id > 0)
							in_it = local_additional_buffers[1 - current_output_buffer_index]->begin();
						else
							in_it = input_buffer_it + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);
						int max_output_size = *(dimension_sizes_it + dimension_id);
						int input_slice_size = *(input_slices_it + dimension_id);

						std::vector<unsigned int> current_output_position(dimension_count, 0);
						for(std::vector<float>::iterator out_it = out_it_base; out_it != out_it_base + output_neuron_count_per_feature_map; ++out_it, ++in_it)
						{
							const std::vector<float>& current_window_weights_list = *(window_weights_list_it + dimension_id);
							float sum = *in_it * current_window_weights_list[0];

							int current_position = static_cast<int>(current_output_position[dimension_id]);
							int dest_forward = current_position;
							int dest_backward = dest_forward;
							for (std::vector<float>::const_iterator it = current_window_weights_list.begin() + 1; it != current_window_weights_list.end(); ++it)
							{
								dest_forward++;
								dest_backward--;
								int dest_forward_actual = (dest_forward < max_output_size) ? dest_forward : (((max_output_size << 1) - 1) - dest_forward);
								int dest_backward_actual = (dest_backward >= 0) ? dest_backward : (-1 - dest_backward);
								int offset_forward = ((dest_forward_actual - current_position) * input_slice_size);
								int offset_backward = ((dest_backward_actual - current_position) * input_slice_size);
								sum += (*(in_it + offset_forward) + *(in_it + offset_backward)) * (*it);
							}

							*out_it = sum;

							// Go to the next output element
							for(unsigned int i = 0; i < dimension_count; ++i)
							{
								if ((++current_output_position[i]) < *(dimension_sizes_it + i))
									break;
								current_output_position[i] = 0;
							}
						}

						current_output_buffer_index = 1 - current_output_buffer_index;
					} // for(unsigned int dimension_id

					// Subtract the gaussian blur
					{
						std::vector<float>::const_iterator original_in_it = input_buffer_it + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);
						std::vector<float>::iterator out_it = output_buffer_it + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);
						std::vector<float>::const_iterator in_it = local_additional_buffers[1 - current_output_buffer_index]->begin();
						for(int i = 0; i < static_cast<int>(input_neuron_count_per_feature_map); ++i)
							*(out_it + i) = *(original_in_it + i) - *(in_it + i);
					}
				}
			} // #pragma parallel

			if (feature_maps_unaffected_count > 0)
			{
				for(unsigned int entry_id = 0; entry_id < updater_count; ++entry_id)
				{
					for(std::vector<unsigned int>::const_iterator it = feature_maps_unaffected.begin(); it != feature_maps_unaffected.end(); ++it)
					{
						unsigned int feature_map_id = *it;
						std::vector<float>::const_iterator original_in_it = input_buffer_it + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);
						std::vector<float>::iterator out_it = output_buffer_it + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);
						std::copy(original_in_it, original_in_it + input_neuron_count_per_feature_map, out_it);
					}
				}
			}
		}

		void local_contrast_subtractive_layer_updater_plain::backprop(
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
			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int output_neuron_count = output_configuration_specific.get_neuron_count();
			const unsigned int output_neuron_count_per_feature_map = output_configuration_specific.get_neuron_count_per_feature_map();
			nnforge_shared_ptr<const local_contrast_subtractive_layer> layer_derived = nnforge_dynamic_pointer_cast<const local_contrast_subtractive_layer>(layer_schema);
			const std::vector<std::vector<float> >& window_weights_list = layer_derived->window_weights_list;
			const std::vector<unsigned int>& feature_maps_affected = layer_derived->feature_maps_affected;
			const unsigned int dimension_count = static_cast<unsigned int>(window_weights_list.size());
			std::vector<unsigned int> input_slices(input_configuration_specific.dimension_sizes.size());
			input_slices[0] = 1;
			for(unsigned int i = 0; i < dimension_count - 1; ++i)
				input_slices[i + 1] = input_slices[i] * input_configuration_specific.dimension_sizes[i];

			const std::vector<unsigned int>::const_iterator dimension_sizes_it = output_configuration_specific.dimension_sizes.begin();
			const unsigned int feature_maps_affected_count = static_cast<unsigned int>(feature_maps_affected.size());
			const std::vector<unsigned int>::const_iterator input_slices_it = input_slices.begin();
			const std::vector<unsigned int>::const_iterator feature_maps_affected_it = feature_maps_affected.begin();
			const std::vector<float>::iterator input_buffer_it = input_errors->begin();
			const std::vector<std::vector<float> >::const_iterator window_weights_list_it = window_weights_list.begin();

			const int total_workload = updater_count * feature_maps_affected_count;
			const int openmp_thread_count = plain_config->openmp_thread_count;
			
			#pragma omp parallel default(none) shared(additional_buffers) num_threads(openmp_thread_count)
			{
				std::vector<additional_buffer_smart_ptr> local_additional_buffers;
				int thread_id = 0;
				#ifdef _OPENMP
				thread_id = omp_get_thread_num();
				#endif

				local_additional_buffers.push_back(additional_buffers[thread_id]);
				if (dimension_count > 1)
					local_additional_buffers.push_back(additional_buffers[openmp_thread_count + thread_id]);

				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int entry_id = workload_id / feature_maps_affected_count;
					int affected_feature_map_id = workload_id - (entry_id * feature_maps_affected_count);

					unsigned int current_output_buffer_index = 0;
					unsigned int feature_map_id = *(feature_maps_affected_it + affected_feature_map_id);
					for(unsigned int dimension_id = 0; dimension_id < dimension_count; ++dimension_id)
					{
						std::vector<float>::iterator out_it_base = local_additional_buffers[current_output_buffer_index]->begin();
						std::vector<float>::const_iterator in_it;
						if (dimension_id > 0)
							in_it = local_additional_buffers[1 - current_output_buffer_index]->begin();
						else
							in_it = input_buffer_it + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);
						int max_output_size = *(dimension_sizes_it + dimension_id);
						int input_slice_size = *(input_slices_it + dimension_id);

						std::vector<unsigned int> current_output_position(dimension_count, 0);
						for(std::vector<float>::iterator out_it = out_it_base; out_it != out_it_base + output_neuron_count_per_feature_map; ++out_it, ++in_it)
						{
							const std::vector<float>& current_window_weights_list = *(window_weights_list_it + dimension_id);
							float sum = *in_it * current_window_weights_list[0];

							int current_position = static_cast<int>(current_output_position[dimension_id]);
							int dest_forward = current_position;
							int dest_backward = dest_forward;
							for (std::vector<float>::const_iterator it = current_window_weights_list.begin() + 1; it != current_window_weights_list.end(); ++it)
							{
								dest_forward++;
								dest_backward--;
								int dest_forward_actual = (dest_forward < max_output_size) ? dest_forward : (((max_output_size << 1) - 1) - dest_forward);
								int dest_backward_actual = (dest_backward >= 0) ? dest_backward : (-1 - dest_backward);
								int offset_forward = ((dest_forward_actual - current_position) * input_slice_size);
								int offset_backward = ((dest_backward_actual - current_position) * input_slice_size);
								sum += (*(in_it + offset_forward) + *(in_it + offset_backward)) * (*it);
							}

							*out_it = sum;

							// Go to the next output element
							for(unsigned int i = 0; i < dimension_count; ++i)
							{
								if ((++current_output_position[i]) < *(dimension_sizes_it + i))
									break;
								current_output_position[i] = 0;
							}
						}

						current_output_buffer_index = 1 - current_output_buffer_index;
					} // for(unsigned int dimension_id

					{
						std::vector<float>::iterator out_it = input_buffer_it + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);
						std::vector<float>::const_iterator in_it = local_additional_buffers[1 - current_output_buffer_index]->begin();
						for(int i = 0; i < static_cast<int>(input_neuron_count_per_feature_map); ++i)
							*(out_it + i) -= *(in_it + i);
					}
				}
			} // #pragma parallel
		}

		std::vector<std::pair<unsigned int, bool> > local_contrast_subtractive_layer_updater_plain::get_elem_count_and_per_entry_flag_additional_buffers(
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			plain_running_configuration_const_smart_ptr plain_config,
			bool backprop_required) const
		{
			std::vector<std::pair<unsigned int, bool> > res;

			nnforge_shared_ptr<const local_contrast_subtractive_layer> layer_derived = nnforge_dynamic_pointer_cast<const local_contrast_subtractive_layer>(layer_schema);
			unsigned int elem_count_per_intermediate_elem = static_cast<unsigned int>(layer_derived->feature_maps_affected.size() * output_configuration_specific.get_neuron_count_per_feature_map());

			for(int i = 0; i < plain_config->openmp_thread_count; ++i)
				res.push_back(std::make_pair(elem_count_per_intermediate_elem, false));
			if (input_configuration_specific.dimension_sizes.size() > 1)
				for(int i = 0; i < plain_config->openmp_thread_count; ++i)
					res.push_back(std::make_pair(elem_count_per_intermediate_elem, false));

			return res;
		}

		bool local_contrast_subtractive_layer_updater_plain::is_in_place_backprop() const
		{
			return true;
		}
	}
}
