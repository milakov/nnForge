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

#include "rectified_linear_layer_tester_plain.h"

#include "../rectified_linear_layer.h"

namespace nnforge
{
	namespace plain
	{
		rectified_linear_layer_tester_plain::rectified_linear_layer_tester_plain()
		{
		}

		rectified_linear_layer_tester_plain::~rectified_linear_layer_tester_plain()
		{
		}

		std::string rectified_linear_layer_tester_plain::get_type_name() const
		{
			return rectified_linear_layer::layer_type_name;
		}

		void rectified_linear_layer_tester_plain::run_forward_propagation(
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
			const int elem_count = static_cast<int>(entry_count * output_configuration_specific.get_neuron_count());
			float * const out_it = *output_buffer;
			const float * const in_it = *input_buffers[0];

			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int i = 0; i < elem_count; ++i)
				*(out_it + i) = std::max<float>(*(in_it + i), 0.0F);
		}

		int rectified_linear_layer_tester_plain::get_input_index_layer_can_write(
			plain_running_configuration::const_ptr plain_config,
			layer::const_ptr layer_schema,
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return 0;
		}
	}
}
