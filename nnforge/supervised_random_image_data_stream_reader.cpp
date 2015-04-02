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

#include "supervised_random_image_data_stream_reader.h"

#include <cstring>

namespace nnforge
{
	supervised_random_image_data_stream_reader::supervised_random_image_data_stream_reader(
		nnforge_shared_ptr<std::istream> input_stream,
		unsigned int original_image_width,
		unsigned int original_image_height,
		unsigned int cropped_image_width,
		unsigned int cropped_image_height,
		unsigned int class_count,
		bool is_color,
		bool is_deterministic)
		: supervised_image_stream_reader(input_stream, original_image_width, original_image_height, false, false, is_color)
		, is_deterministic(is_deterministic)
		, gen(rnd::get_random_generator())
	{
		input_configuration.feature_map_count = is_color ? 3 : 1;
		input_configuration.dimension_sizes.push_back(cropped_image_width);
		input_configuration.dimension_sizes.push_back(cropped_image_height);
		input_neuron_count = input_configuration.get_neuron_count();

		output_configuration.feature_map_count = class_count;
		output_configuration.dimension_sizes.push_back(1);
		output_configuration.dimension_sizes.push_back(1);
		output_neuron_count = output_configuration.get_neuron_count();
	}

	supervised_random_image_data_stream_reader::~supervised_random_image_data_stream_reader()
	{
	}

	bool supervised_random_image_data_stream_reader::read(
		void * input_neurons,
		float * output_neurons)
	{
		cv::Mat image;
		unsigned int class_id;

		std::vector<unsigned char> output_data;
		bool res = read_image(input_neurons ? &image : 0, output_neurons ? &output_data : 0);
		if (!res)
			return false;

		if (output_neurons)
			class_id = *reinterpret_cast<int *>(&output_data[0]);

		if (input_neurons)
		{
			unsigned int start_src_x;
			if (is_deterministic)
			{
				start_src_x = (image.cols - input_configuration.dimension_sizes[0]) / 2;
			}
			else
			{
				unsigned int max_start_src_x = image.cols - input_configuration.dimension_sizes[0];
				if (max_start_src_x > 0)
				{
					nnforge_uniform_int_distribution<unsigned int> dist(0, max_start_src_x);
					start_src_x = dist(gen);
				}
			}

			unsigned int start_src_y;
			if (is_deterministic)
			{
				start_src_y = (image.rows - input_configuration.dimension_sizes[1]) / 2;
			}
			else
			{
				unsigned int max_start_src_y = image.rows - input_configuration.dimension_sizes[1];
				if (max_start_src_y > 0)
				{
					nnforge_uniform_int_distribution<unsigned int> dist(0, max_start_src_y);
					start_src_y = dist(gen);
				}
			}

			unsigned int src_y = start_src_y;
			for(unsigned int dst_y = 0; dst_y < input_configuration.dimension_sizes[1]; ++dst_y, ++src_y)
			{
				if (is_color)
				{
					unsigned char * dst_ptr_r = ((unsigned char *)input_neurons) + dst_y * input_configuration.dimension_sizes[0];
					unsigned char * dst_ptr_g = dst_ptr_r + input_configuration.dimension_sizes[0] * input_configuration.dimension_sizes[1];
					unsigned char * dst_ptr_b = dst_ptr_r + 2 * input_configuration.dimension_sizes[0] * input_configuration.dimension_sizes[1];
					const cv::Vec3b * src_ptr = image.ptr<cv::Vec3b>(src_y) + start_src_x;
					for(unsigned int dst_x = 0; dst_x < input_configuration.dimension_sizes[0]; ++dst_x)
					{
						cv::Vec3b val = src_ptr[dst_x];
						dst_ptr_r[dst_x] = val[2];
						dst_ptr_g[dst_x] = val[1];
						dst_ptr_b[dst_x] = val[0];
					}
				}
				else
				{
					unsigned char * dst_ptr = ((unsigned char *)input_neurons) + dst_y * input_configuration.dimension_sizes[0];
					const unsigned char * src_ptr = image.ptr<unsigned char>(src_y) + start_src_x;
					memcpy(dst_ptr, src_ptr, input_configuration.dimension_sizes[0]);
				}
			}
		}

		if (output_neurons)
		{
			std::fill_n(output_neurons, output_neuron_count, 0.0F);
			output_neurons[class_id] = 1.0F;
		}

		return true;
	}
}
