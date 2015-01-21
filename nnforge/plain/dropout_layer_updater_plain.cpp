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

#include "dropout_layer_updater_plain.h"

#include "../dropout_layer.h"
#include "../neural_network_exception.h"
#include "../nn_types.h"

namespace nnforge
{
	namespace plain
	{
		dropout_layer_updater_plain::dropout_layer_updater_plain()
			: gen(rnd::get_random_generator())
		{
		}

		dropout_layer_updater_plain::~dropout_layer_updater_plain()
		{
		}

		const boost::uuids::uuid& dropout_layer_updater_plain::get_uuid() const
		{
			return dropout_layer::layer_guid;
		}

		void dropout_layer_updater_plain::test(
			const_additional_buffer_smart_ptr input_buffer,
			additional_buffer_smart_ptr output_buffer,
			std::vector<additional_buffer_smart_ptr>& additional_buffers,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const_layer_data_custom_smart_ptr data_custom,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int updater_count,
			unsigned int offset_input_entry_id,
			bool force_deterministic) const
		{
			if (offset_input_entry_id > 0)
				throw neural_network_exception("dropout_layer_updater_plain is not able to run using offset");

			if (force_deterministic)
			{
				memcpy(&(output_buffer->at(0)), &(input_buffer->at(0)), input_configuration_specific.get_neuron_count() * updater_count * sizeof(float));
			}
			else
			{
				const std::vector<float>::const_iterator in_it_global = input_buffer->begin();
				const std::vector<float>::iterator out_it_global = output_buffer->begin();
				unsigned char * keep_elem_ptr = reinterpret_cast<unsigned char *>(&(additional_buffers[0]->at(0)));

				nnforge_shared_ptr<const dropout_layer> layer_derived = nnforge_dynamic_pointer_cast<const dropout_layer>(layer_schema);
				const float dropout_rate = layer_derived->dropout_rate;
				const float keep_rate = 1.0F - dropout_rate;
				const float mult = 1.0F / keep_rate;

				const int total_workload = input_configuration_specific.get_neuron_count() * updater_count;

				nnforge_uniform_real_distribution<float> dist(0.0F, 1.0F);

				for(int i = 0; i < total_workload; ++i)
					keep_elem_ptr[i] = (dist(gen) <= keep_rate ? (unsigned char)1 : (unsigned char)0);

				#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count) shared(keep_elem_ptr)
				{
					#pragma omp for schedule(guided)
					for(int workload_id = 0; workload_id < total_workload; ++workload_id)
					{
						int elem_id = workload_id;
						*(out_it_global + elem_id) = *(in_it_global + elem_id) * (keep_elem_ptr[elem_id] ? mult : 0.0F);
					}
				}
			}
		}

		void dropout_layer_updater_plain::backprop(
			additional_buffer_smart_ptr input_errors,
			const_additional_buffer_smart_ptr input_neurons,
			const_additional_buffer_smart_ptr output_errors,
			const_additional_buffer_smart_ptr output_neurons,
			std::vector<additional_buffer_smart_ptr>& additional_buffers,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const_layer_data_custom_smart_ptr data_custom,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int updater_count,
			bool force_deterministic) const
		{
			if (force_deterministic)
				return;

			const std::vector<float>::iterator in_err_it_global = input_errors->begin();
			unsigned char * keep_elem_ptr = reinterpret_cast<unsigned char *>(&(additional_buffers[0]->at(0)));

			nnforge_shared_ptr<const dropout_layer> layer_derived = nnforge_dynamic_pointer_cast<const dropout_layer>(layer_schema);
			const float dropout_rate = layer_derived->dropout_rate;
			const float keep_rate = 1.0F - dropout_rate;
			const float mult = 1.0F / keep_rate;

			const int total_workload = input_configuration_specific.get_neuron_count() * updater_count;

			#pragma omp parallel default(none) num_threads(plain_config->openmp_thread_count) shared(keep_elem_ptr)
			{
				#pragma omp for schedule(guided)
				for(int workload_id = 0; workload_id < total_workload; ++workload_id)
				{
					int elem_id = workload_id;
					*(in_err_it_global + elem_id) *= (keep_elem_ptr[elem_id] ? mult : 0.0F);
				}
			}
		}

		std::vector<std::pair<unsigned int, bool> > dropout_layer_updater_plain::get_elem_count_and_per_entry_flag_additional_buffers(
			const_layer_smart_ptr layer_schema,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			plain_running_configuration_const_smart_ptr plain_config,
			bool backprop_required) const
		{
			std::vector<std::pair<unsigned int, bool> > res;

			res.push_back(std::make_pair<unsigned int, bool>((output_configuration_specific.get_neuron_count() + 3) / 4, true));

			return res;
		}

		bool dropout_layer_updater_plain::is_in_place_backprop() const
		{
			return true;
		}
	}
}
