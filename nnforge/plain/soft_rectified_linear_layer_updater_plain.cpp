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

#include "soft_rectified_linear_layer_updater_plain.h"

#include "../soft_rectified_linear_layer.h"
#include "../neural_network_exception.h"

namespace nnforge
{
	namespace plain
	{
		soft_rectified_linear_layer_updater_plain::soft_rectified_linear_layer_updater_plain()
		{
		}

		soft_rectified_linear_layer_updater_plain::~soft_rectified_linear_layer_updater_plain()
		{
		}

		const boost::uuids::uuid& soft_rectified_linear_layer_updater_plain::get_uuid() const
		{
			return soft_rectified_linear_layer::layer_guid;
		}

		void soft_rectified_linear_layer_updater_plain::test(
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
			if (offset_input_entry_id >= 0)
				throw neural_network_exception("hyperbolic_tangent_layer_updater_plain is not able to run using offset");

			const int elem_count = static_cast<int>(updater_count * input_configuration_specific.get_neuron_count());
			const std::vector<float>::const_iterator in_it = input_buffer->begin();
			const std::vector<float>::iterator out_it = output_buffer->begin();

			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int i = 0; i < elem_count; ++i)
				*(out_it + i) = logf(expf(*(in_it + i)) + 1.0F);
		}

		void soft_rectified_linear_layer_updater_plain::backprop(
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
			const int elem_count = static_cast<int>(updater_count * input_configuration_specific.get_neuron_count());
			const std::vector<float>::iterator in_err_it = input_errors->begin();
			const std::vector<float>::const_iterator out_it = output_neurons->begin();

			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int i = 0; i < elem_count; ++i)
			{
				float out_neuron = *(out_it + i);
				float val = expf(out_neuron);
				float der1st = (val - 1.0F) / val;
				*(in_err_it + i) *= der1st;
			}
		}

		bool soft_rectified_linear_layer_updater_plain::is_in_place_backprop() const
		{
			return true;
		}
	}
}
