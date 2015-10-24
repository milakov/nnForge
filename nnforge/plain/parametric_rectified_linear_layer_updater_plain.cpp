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

#include "parametric_rectified_linear_layer_updater_plain.h"

#include "../parametric_rectified_linear_layer.h"
#include "../neural_network_exception.h"

namespace nnforge
{
	namespace plain
	{
		parametric_rectified_linear_layer_updater_plain::parametric_rectified_linear_layer_updater_plain()
		{
		}

		parametric_rectified_linear_layer_updater_plain::~parametric_rectified_linear_layer_updater_plain()
		{
		}

		const boost::uuids::uuid& parametric_rectified_linear_layer_updater_plain::get_uuid() const
		{
			return parametric_rectified_linear_layer::layer_guid;
		}

		void parametric_rectified_linear_layer_updater_plain::test(
			const_additional_buffer_smart_ptr input_buffer,
			additional_buffer_smart_ptr output_buffer,
			std::vector<additional_buffer_smart_ptr>& additional_buffers,
			plain_running_configuration::const_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const_layer_data_custom_smart_ptr data_custom,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int updater_count,
			unsigned int offset_input_entry_id,
			bool force_deterministic) const
		{
			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int feature_map_count = input_configuration_specific.feature_map_count;

			const int total_workload = static_cast<int>(updater_count * feature_map_count);
			const std::vector<float>::const_iterator in_it = input_buffer->begin() + input_neuron_count * offset_input_entry_id;
			const std::vector<float>::iterator out_it = output_buffer->begin();
			const std::vector<float>::const_iterator weights = (*data)[0].begin();

			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int workload_id = 0; workload_id < total_workload; ++workload_id)
			{
				int entry_id = workload_id / feature_map_count;
				int feature_map_id = workload_id - entry_id * feature_map_count;

				float a = weights[feature_map_id];

				std::vector<float>::const_iterator current_in_it = in_it + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);
				std::vector<float>::const_iterator current_in_it_end = current_in_it + input_neuron_count_per_feature_map;
				std::vector<float>::iterator current_out_it = out_it + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);

				for(; current_in_it != current_in_it_end; ++current_in_it, ++current_out_it)
				{
					float input_val = *current_in_it;
					float output_val = input_val * (input_val >= 0.0F ? 1.0F : a);
					*current_out_it = output_val;
				}
			}
		}

		void parametric_rectified_linear_layer_updater_plain::backprop(
			additional_buffer_smart_ptr input_errors,
			const_additional_buffer_smart_ptr input_neurons,
			const_additional_buffer_smart_ptr output_errors,
			const_additional_buffer_smart_ptr output_neurons,
			std::vector<additional_buffer_smart_ptr>& additional_buffers,
			plain_running_configuration::const_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const_layer_data_custom_smart_ptr data_custom,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int updater_count,
			bool force_deterministic) const
		{
			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int feature_map_count = input_configuration_specific.feature_map_count;

			const int total_workload = static_cast<int>(updater_count * feature_map_count);
			const std::vector<float>::const_iterator in_neurons_it = input_neurons->begin();
			const std::vector<float>::iterator err_it = input_errors->begin();
			const std::vector<float>::const_iterator weights = (*data)[0].begin();

			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int workload_id = 0; workload_id < total_workload; ++workload_id)
			{
				int entry_id = workload_id / feature_map_count;
				int feature_map_id = workload_id - entry_id * feature_map_count;

				float a = weights[feature_map_id];

				std::vector<float>::const_iterator current_in_neurons_it = in_neurons_it + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);
				std::vector<float>::const_iterator current_in_neurons_it_end = current_in_neurons_it + input_neuron_count_per_feature_map;
				std::vector<float>::iterator current_err_it = err_it + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);

				for(; current_in_neurons_it != current_in_neurons_it_end; ++current_in_neurons_it, ++current_err_it)
				{
					float output_err = *current_err_it;
					float input_val = *current_in_neurons_it;
					float input_err = output_err * (input_val >= 0.0F ? 1.0F : a);
					*current_err_it = input_err;
				}
			}
		}

		void parametric_rectified_linear_layer_updater_plain::update_weights(
			const_additional_buffer_smart_ptr input_neurons,
			const_additional_buffer_smart_ptr output_errors,
			std::vector<additional_buffer_smart_ptr>& additional_buffers,
			layer_data_smart_ptr gradient,
			const_layer_data_custom_smart_ptr data_custom,
			plain_running_configuration::const_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int updater_count,
			unsigned int offset_input_entry_id,
			bool force_deterministic) const
		{
			const unsigned int input_neuron_count = input_configuration_specific.get_neuron_count();
			const unsigned int input_neuron_count_per_feature_map = input_configuration_specific.get_neuron_count_per_feature_map();
			const unsigned int feature_map_count = input_configuration_specific.feature_map_count;

			const std::vector<float>::const_iterator in_neurons_it = input_neurons->begin() + input_neuron_count * offset_input_entry_id;
			const std::vector<float>::const_iterator err_it = output_errors->begin();
			const std::vector<float>::iterator gradients = (*gradient)[0].begin();

			const int total_workload = feature_map_count;
			const int const_updater_count = updater_count;

			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int workload_id = 0; workload_id < total_workload; ++workload_id)
			{
				int feature_map_id = workload_id;

				float sum = 0.0F;
				for(int entry_id = 0; entry_id < const_updater_count; ++entry_id)
				{
					std::vector<float>::const_iterator current_in_neurons_it = in_neurons_it + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);
					std::vector<float>::const_iterator current_in_neurons_it_end = current_in_neurons_it + input_neuron_count_per_feature_map;
					std::vector<float>::const_iterator current_err_it = err_it + (entry_id * input_neuron_count) + (feature_map_id * input_neuron_count_per_feature_map);

					float local_sum = 0.0F;
					for(; current_in_neurons_it != current_in_neurons_it_end; ++current_in_neurons_it, ++current_err_it)
					{
						float output_err = *current_err_it;
						float input_val = *current_in_neurons_it;
						float gr = output_err * (input_val >= 0.0F ? 0.0F : input_val);
						local_sum += gr;
					}

					sum += local_sum;
				}

				*(gradients + feature_map_id) += sum;
			}
		}

		bool parametric_rectified_linear_layer_updater_plain::is_in_place_backprop() const
		{
			return true;
		}
	}
}
