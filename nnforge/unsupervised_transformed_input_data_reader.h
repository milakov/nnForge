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

#include "unsupervised_data_reader.h"
#include "data_transformer.h"

#include <memory>

namespace nnforge
{
	class unsupervised_transformed_input_data_reader : public unsupervised_data_reader
	{
	public:
		unsupervised_transformed_input_data_reader(
			unsupervised_data_reader_smart_ptr original_reader,
			data_transformer_smart_ptr transformer);

		virtual ~unsupervised_transformed_input_data_reader();

		// The method should return true in case entry is read and false if there is no more entries available (and no entry is read in this case)
		// If any parameter is null the method should just discard corresponding data
		virtual bool read(void * input_elems);

		virtual void reset();

		virtual layer_configuration_specific get_input_configuration() const;

		virtual neuron_data_type::input_type get_input_type() const;

		virtual void set_max_entries_to_read(unsigned int max_entries_to_read);

	protected:
		unsupervised_transformed_input_data_reader();

		virtual unsigned int get_actual_entry_count() const;

	protected:
		unsupervised_data_reader_smart_ptr original_reader;
		data_transformer_smart_ptr transformer;

		std::vector<unsigned char> buf;
		void * local_input_ptr;
		unsigned int current_sample_id;
		unsigned int transformer_sample_count;
	};
}
