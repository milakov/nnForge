/*
 *  Copyright 2011-2016 Maxim Milakov
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

#include "elastic_deformation_2d_data_transformer.h"

#include "neural_network_exception.h"
#include "data_transformer_util.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	elastic_deformation_2d_data_transformer::elastic_deformation_2d_data_transformer(
		float sigma,
		float alpha,
		float border_value)
		: alpha(alpha)
		, sigma(sigma)
		, border_value(border_value)
		, gen(rnd::get_random_generator())
		, displacement_distribution(std::uniform_real_distribution<float>(-1.0F, 1.0F))
	{
	}

	void elastic_deformation_2d_data_transformer::transform(
		const float * data,
		float * data_transformed,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (original_config.dimension_sizes.size() < 2)
			throw neural_network_exception((boost::format("intensity_2d_data_transformer is processing at least 2d data, data is passed with number of dimensions %1%") % original_config.dimension_sizes.size()).str());

		int ksize = static_cast<int>((sigma - 0.8F) * 3.0F + 1.0F) * 2 + 1;

		cv::Mat1f x_disp(original_config.dimension_sizes[1], original_config.dimension_sizes[0]);
		cv::Mat1f y_disp(original_config.dimension_sizes[1], original_config.dimension_sizes[0]);

		{
			std::lock_guard<std::mutex> lock(gen_stream_mutex);

			for(int row_id = 0; row_id < x_disp.rows; ++row_id)
			{
				float * row_ptr = x_disp.ptr<float>(row_id);
				for(int column_id = 0; column_id < x_disp.cols; ++column_id)
					row_ptr[column_id] = displacement_distribution(gen);
			}

			for(int row_id = 0; row_id < y_disp.rows; ++row_id)
			{
				float * row_ptr = y_disp.ptr<float>(row_id);
				for(int column_id = 0; column_id < y_disp.cols; ++column_id)
					row_ptr[column_id] = displacement_distribution(gen);
			}
		}

		smooth(x_disp, ksize, sigma, alpha, true);
		smooth(y_disp, ksize, sigma, alpha, false);

		unsigned int neuron_count_per_image = original_config.dimension_sizes[0] * original_config.dimension_sizes[1];
		unsigned int image_count = original_config.get_neuron_count() / neuron_count_per_image;
		for(unsigned int image_id = 0; image_id < image_count; ++image_id)
		{
			cv::Mat1f image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), const_cast<float *>(data) + (image_id * neuron_count_per_image));
			cv::Mat1f image_dst(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), data_transformed + (image_id * neuron_count_per_image));

			cv::remap(image, image_dst, x_disp, y_disp, cv::INTER_LINEAR, cv::BORDER_CONSTANT, border_value);
		}
	} 

	void elastic_deformation_2d_data_transformer::smooth(
		cv::Mat1f disp,
		int ksize,
		float sigma,
		float alpha,
		bool is_x)
	{
		cv::GaussianBlur(disp, disp, cv::Size(ksize, ksize), sigma, sigma, cv::BORDER_REFLECT_101);
		float sum_of_squared = 0.0F;
		for(int row_id = 0; row_id < disp.rows; ++row_id)
		{
			float * row_ptr = disp.ptr<float>(row_id);
			float sum_of_squared_local = 0.0F;
			for(int column_id = 0; column_id < disp.cols; ++column_id)
			{
				float val = row_ptr[column_id];
				sum_of_squared_local += val * val;
			}
			sum_of_squared += sum_of_squared_local;
		}
		float disp_norm = sqrtf(sum_of_squared);
		float mult = alpha / disp_norm;

		if (is_x)
		{
			for(int row_id = 0; row_id < disp.rows; ++row_id)
			{
				float * row_ptr = disp.ptr<float>(row_id);
				for(int column_id = 0; column_id < disp.cols; ++column_id)
				{
					float column_id_f = static_cast<float>(column_id);
					row_ptr[column_id] = row_ptr[column_id] * mult + column_id_f;
				}
			}
		}
		else
		{
			for(int row_id = 0; row_id < disp.rows; ++row_id)
			{
				float row_id_f = static_cast<float>(row_id);
				float * row_ptr = disp.ptr<float>(row_id);
				for(int column_id = 0; column_id < disp.cols; ++column_id)
				{
					row_ptr[column_id] = row_ptr[column_id] * mult + row_id_f;
				}
			}
		}
	}
}
