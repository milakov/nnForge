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

#pragma once

#include "unsupervised_data_stream_schema.h"
#include "layer_configuration_specific.h"
#include "neuron_data_type.h"
#include "nn_types.h"

#include <vector>
#include <ostream>

namespace nnforge
{
	class unsupervised_data_stream_writer
	{
	public:
		// The constructor modifies output_stream to throw exceptions in case of failure
		// The stream should be created with std::ios_base::binary flag
		unsupervised_data_stream_writer(
			nnforge_shared_ptr<std::ostream> output_stream,
			const layer_configuration_specific& input_configuration);

		virtual ~unsupervised_data_stream_writer();

		void write(
			neuron_data_type::input_type type_code,
			const void * input_neurons);

		void write(const float * input_neurons);

		void write(const unsigned char * input_neurons);

	private:
		nnforge_shared_ptr<std::ostream> out_stream;
		unsigned int input_neuron_count;

		std::ostream::pos_type type_code_pos;
		neuron_data_type::input_type type_code;
		size_t input_elem_size;

		std::ostream::pos_type entry_count_pos;
		unsigned int entry_count;

	private:
		unsupervised_data_stream_writer(const unsupervised_data_stream_writer&);
		unsupervised_data_stream_writer& operator =(const unsupervised_data_stream_writer&);
	};

	typedef nnforge_shared_ptr<unsupervised_data_stream_writer> unsupervised_data_stream_writer_smart_ptr;
}
