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

#include "sigmoid_layer_tester_plain.h"

#include "../sigmoid_layer.h"
#include "../nn_types.h"

namespace nnforge
{
	namespace plain
	{
		sigmoid_layer_tester_plain::sigmoid_layer_tester_plain()
		{
		}

		sigmoid_layer_tester_plain::~sigmoid_layer_tester_plain()
		{
		}

		const boost::uuids::uuid& sigmoid_layer_tester_plain::get_uuid() const
		{
			return sigmoid_layer::layer_guid;
		}

		void sigmoid_layer_tester_plain::test(
			additional_buffer_smart_ptr input_buffer,
			additional_buffer_set& additional_buffers,
			plain_running_configuration_const_smart_ptr plain_config,
			const_layer_smart_ptr layer_schema,
			const_layer_data_smart_ptr data,
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int entry_count) const
		{
			const int elem_count = static_cast<int>(entry_count * input_configuration_specific.get_neuron_count());
			const std::vector<float>::iterator in_it = input_buffer->begin();

			#pragma omp parallel for default(none) schedule(guided) num_threads(plain_config->openmp_thread_count)
			for(int i = 0; i < elem_count; ++i)
			{
				float inp = *(in_it + i);
				float res = 1.0F / (expf(-inp) + 1.0F);
				*(in_it + i) = res;
			}
		}
	}
}
