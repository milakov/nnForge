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

namespace nnforge
{
	class supervised_image_data_sampler_stream_reader : public supervised_image_stream_reader
	{
	public:
		// The constructor modifies input_stream to throw exceptions in case of failure
		supervised_image_data_sampler_stream_reader(
			nnforge_shared_ptr<std::istream> input_stream,
			unsigned int original_image_width,
			unsigned int original_image_height,
			unsigned int cropped_image_width,
			unsigned int cropped_image_height,
			unsigned int class_count,
			bool fit_image,
			bool is_color = true,
			unsigned char backfill_intensity = 128,
			const std::vector<std::pair<float, float> >& position_list = std::vector<std::pair<float, float> >(1, std::make_pair(0.5F, 0.5F)));

		virtual ~supervised_image_data_sampler_stream_reader();

		virtual bool read(
			void * input_neurons,
			float * output_neurons);

		virtual bool raw_read(std::vector<unsigned char>& all_elems);

		virtual layer_configuration_specific get_input_configuration() const
		{
			return input_configuration;
		}

		virtual layer_configuration_specific get_output_configuration() const
		{
			return output_configuration;
		}

		virtual void next_epoch();

		virtual void reset();

		virtual unsigned int get_entry_count() const;

		virtual void rewind(unsigned int entry_id);

		virtual unsigned int get_sample_count() const;

	protected:
		layer_configuration_specific input_configuration;
		layer_configuration_specific output_configuration;
		unsigned int input_neuron_count;
		unsigned int output_neuron_count;
		unsigned char backfill_intensity;
		const std::vector<std::pair<float, float> > position_list;

		unsigned int current_sample_id;
		cv::Mat image;
		unsigned int class_id;
	};
}
