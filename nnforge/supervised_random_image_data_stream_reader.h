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

#pragma once

#include "supervised_image_stream_reader.h"
#include "rnd.h"

namespace nnforge
{
	class supervised_random_image_data_stream_reader : public supervised_image_stream_reader
	{
	public:
		// The constructor modifies input_stream to throw exceptions in case of failure
		supervised_random_image_data_stream_reader(
			nnforge_shared_ptr<std::istream> input_stream,
			unsigned int original_image_width,
			unsigned int original_image_height,
			unsigned int cropped_image_width,
			unsigned int cropped_image_height,
			unsigned int class_count,
			bool is_color = true,
			bool is_deterministic = false);

		virtual ~supervised_random_image_data_stream_reader();

		virtual bool read(
			void * input_neurons,
			float * output_neurons);

		virtual layer_configuration_specific get_input_configuration() const
		{
			return input_configuration;
		}

		virtual layer_configuration_specific get_output_configuration() const
		{
			return output_configuration;
		}

	protected:
		layer_configuration_specific input_configuration;
		layer_configuration_specific output_configuration;
		unsigned int input_neuron_count;
		unsigned int output_neuron_count;
		bool is_deterministic;
		
		random_generator gen;
	};
}
