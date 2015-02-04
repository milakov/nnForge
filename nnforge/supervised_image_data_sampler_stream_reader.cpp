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

#include "supervised_image_data_sampler_stream_reader.h"

#include "neural_network_exception.h"

#include <cstring>

namespace nnforge
{
	supervised_image_data_sampler_stream_reader::supervised_image_data_sampler_stream_reader(
		nnforge_shared_ptr<std::istream> input_stream,
		unsigned int original_image_width,
		unsigned int original_image_height,
		unsigned int cropped_image_width,
		unsigned int cropped_image_height,
		unsigned int class_count,
		bool fit_image,
		bool is_color,
		unsigned char backfill_intensity,
		const std::vector<std::pair<float, float> >& position_list)
		: supervised_image_stream_reader(input_stream, original_image_width, original_image_height, fit_image, is_color)
		, position_list(position_list)
		, backfill_intensity(backfill_intensity)
		, current_sample_id(0)
	{
		if (fit_image && ((cropped_image_width != original_image_width) || (cropped_image_height != original_image_height)))
			throw neural_network_exception("supervised_center_image_data_stream_reader: When in fit_image mode, cropped and original images sizes should match");

		input_configuration.feature_map_count = is_color ? 3 : 1;
		input_configuration.dimension_sizes.push_back(cropped_image_width);
		input_configuration.dimension_sizes.push_back(cropped_image_height);
		input_neuron_count = input_configuration.get_neuron_count();

		output_configuration.feature_map_count = class_count;
		output_configuration.dimension_sizes.push_back(1);
		output_configuration.dimension_sizes.push_back(1);
		output_neuron_count = output_configuration.get_neuron_count();
	}

	supervised_image_data_sampler_stream_reader::~supervised_image_data_sampler_stream_reader()
	{
	}

	bool supervised_image_data_sampler_stream_reader::read(
		void * input_neurons,
		float * output_neurons)
	{
		bool read = true;
		if (current_sample_id == 0)
		{
			read = read_image(input_neurons ? &image : 0, output_neurons ? &class_id : 0);
		}
		if (!read)
			return false;

		if (input_neurons)
		{
			if (fit_into_target)
			{
				memset(input_neurons, backfill_intensity, input_neuron_count);
				unsigned int start_dst_x = static_cast<unsigned int>((input_configuration.dimension_sizes[0] - image.cols) * position_list[current_sample_id].first + 0.5F);
				unsigned int start_dst_y = static_cast<unsigned int>((input_configuration.dimension_sizes[1] - image.rows) * position_list[current_sample_id].second + 0.5F);

				unsigned int dst_y = start_dst_y;
				for(unsigned int src_y = 0; src_y < static_cast<unsigned int>(image.rows); ++src_y, ++dst_y)
				{
					if (is_color)
					{
						unsigned char * dst_ptr_r = ((unsigned char *)input_neurons) + dst_y * input_configuration.dimension_sizes[0] + start_dst_x;
						unsigned char * dst_ptr_g = dst_ptr_r + input_configuration.dimension_sizes[0] * input_configuration.dimension_sizes[1];
						unsigned char * dst_ptr_b = dst_ptr_r + 2 * input_configuration.dimension_sizes[0] * input_configuration.dimension_sizes[1];
						const cv::Vec3b * src_ptr = image.ptr<cv::Vec3b>(src_y);
						for(unsigned int src_x = 0; src_x < static_cast<unsigned int>(image.cols); ++src_x)
						{
							cv::Vec3b val = src_ptr[src_x];
							dst_ptr_r[src_x] = val[2];
							dst_ptr_g[src_x] = val[1];
							dst_ptr_b[src_x] = val[0];
						}
					}
					else
					{
						unsigned char * dst_ptr = ((unsigned char *)input_neurons) + dst_y * input_configuration.dimension_sizes[0] + start_dst_x;
						const unsigned char * src_ptr = image.ptr<unsigned char>(src_y);
						memcpy(dst_ptr, src_ptr, image.cols);
					}
				}
			}
			else
			{
				unsigned int start_src_x = static_cast<unsigned int>((image.cols - input_configuration.dimension_sizes[0]) * position_list[current_sample_id].first + 0.5F);
				unsigned int start_src_y = static_cast<unsigned int>((image.rows - input_configuration.dimension_sizes[1]) * position_list[current_sample_id].second + 0.5F);
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
		}

		if (output_neurons)
		{
			std::fill_n(output_neurons, output_neuron_count, 0.0F);
			output_neurons[class_id] = 1.0F;
		}

		current_sample_id = (current_sample_id + 1) % position_list.size();

		return true;
	}

	void supervised_image_data_sampler_stream_reader::next_epoch()
	{
		current_sample_id = 0;
		supervised_image_stream_reader::next_epoch();
	}

	void supervised_image_data_sampler_stream_reader::reset()
	{
		current_sample_id = 0;
		supervised_image_stream_reader::reset();
	}

	unsigned int supervised_image_data_sampler_stream_reader::get_entry_count() const
	{
		return supervised_image_stream_reader::get_entry_count() * static_cast<unsigned int>(position_list.size());
	}

	void supervised_image_data_sampler_stream_reader::rewind(unsigned int entry_id)
	{
		if (entry_id % position_list.size())
			throw std::runtime_error("rewind is only partlially implemented for supervised_image_data_sampler_stream_reader");

		supervised_image_stream_reader::rewind(entry_id / static_cast<unsigned int>(position_list.size()));
	}

	bool supervised_image_data_sampler_stream_reader::raw_read(std::vector<unsigned char>& all_elems)
	{
		if (position_list.size() != 1)
			throw std::runtime_error("raw_read is not implemented for supervised_image_data_sampler_stream_reader with non-unit sampling");

		return supervised_image_stream_reader::raw_read(all_elems);
	}

	unsigned int supervised_image_data_sampler_stream_reader::get_sample_count() const
	{
		return static_cast<unsigned int>(position_list.size()) * supervised_image_stream_reader::get_sample_count();
	}
}
