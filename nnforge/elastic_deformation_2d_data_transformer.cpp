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
		unsigned char border_value)
		: alpha(alpha)
		, sigma(sigma)
		, border_value(border_value)
		, gen(rnd::get_random_generator())
		, displacement_distribution(nnforge_uniform_real_distribution<float>(-1.0F, 1.0F))
		, work(io_service)
	{
		for(unsigned int i = 0; i < 2; ++i)
			threadpool.create_thread(boost::bind(&boost::asio::io_service::run, &io_service));
	}

	elastic_deformation_2d_data_transformer::~elastic_deformation_2d_data_transformer()
	{
		io_service.stop();
		threadpool.join_all();
	}

	void elastic_deformation_2d_data_transformer::transform(
		const void * data,
		void * data_transformed,
		neuron_data_type::input_type type,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		if (type != neuron_data_type::type_byte)
			throw neural_network_exception("intensity_2d_data_transformer is implemented for data stored as bytes only");

		if (original_config.dimension_sizes.size() < 2)
			throw neural_network_exception((boost::format("intensity_2d_data_transformer is processing at least 2d data, data is passed with number of dimensions %1%") % original_config.dimension_sizes.size()).str());

		int ksize = static_cast<int>((sigma - 0.8F) * 3.0F + 1.0F) * 2 + 1;

		cv::Mat1f x_disp(original_config.dimension_sizes[1], original_config.dimension_sizes[0]);
		for(int row_id = 0; row_id < x_disp.rows; ++row_id)
		{
			float * row_ptr = x_disp.ptr<float>(row_id);
			for(int column_id = 0; column_id < x_disp.cols; ++column_id)
				row_ptr[column_id] = displacement_distribution(gen);
		}
		nnforge_shared_ptr<smooth_info> info_x(new smooth_info());
		info_x->disp = x_disp;
		info_x->is_x = true;
		info_x->ksize = ksize;
		info_x->sigma = sigma;
		info_x->alpha = alpha;
		io_service.post(boost::bind(smooth_worker, info_x));

		cv::Mat1f y_disp(original_config.dimension_sizes[1], original_config.dimension_sizes[0]);
		for(int row_id = 0; row_id < y_disp.rows; ++row_id)
		{
			float * row_ptr = y_disp.ptr<float>(row_id);
			for(int column_id = 0; column_id < y_disp.cols; ++column_id)
				row_ptr[column_id] = displacement_distribution(gen);
		}
		nnforge_shared_ptr<smooth_info> info_y(new smooth_info());
		info_y->disp = y_disp;
		info_y->is_x = false;
		info_y->ksize = ksize;
		info_y->sigma = sigma;
		info_y->alpha = alpha;
		io_service.post(boost::bind(smooth_worker, info_y));

		{
			boost::unique_lock<boost::mutex> lock(info_x->is_ready_mutex);
			while (!info_x->is_ready)
				info_x->is_ready_condition.wait(lock);
		}

		{
			boost::unique_lock<boost::mutex> lock(info_y->is_ready_mutex);
			while (!info_y->is_ready)
				info_y->is_ready_condition.wait(lock);
		}

		unsigned int neuron_count_per_image = original_config.dimension_sizes[0] * original_config.dimension_sizes[1];
		unsigned int image_count = original_config.get_neuron_count() / neuron_count_per_image;
		for(unsigned int image_id = 0; image_id < image_count; ++image_id)
		{
			cv::Mat1b image(static_cast<int>(original_config.dimension_sizes[1]), static_cast<int>(original_config.dimension_sizes[0]), static_cast<unsigned char *>(data_transformed) + (image_id * neuron_count_per_image));
			cv::Mat1b image_cloned = image.clone();

			cv::remap(image_cloned, image, x_disp, y_disp, cv::INTER_LINEAR, cv::BORDER_CONSTANT, border_value);
		}
	} 

 	bool elastic_deformation_2d_data_transformer::is_deterministic() const
	{
		return false;
	}

	elastic_deformation_2d_data_transformer::smooth_info::smooth_info()
		: is_ready(false)
	{
	}

	void elastic_deformation_2d_data_transformer::smooth_worker(nnforge_shared_ptr<smooth_info> info)
	{
		cv::GaussianBlur(info->disp, info->disp, cv::Size(info->ksize, info->ksize), info->sigma, info->sigma, cv::BORDER_REFLECT_101);
		float sum_of_squared = 0.0F;
		for(int row_id = 0; row_id < info->disp.rows; ++row_id)
		{
			float * row_ptr = info->disp.ptr<float>(row_id);
			float sum_of_squared_local = 0.0F;
			for(int column_id = 0; column_id < info->disp.cols; ++column_id)
			{
				float val = row_ptr[column_id];
				sum_of_squared_local += val * val;
			}
			sum_of_squared += sum_of_squared_local;
		}
		float disp_norm = sqrtf(sum_of_squared);
		float mult = info->alpha / disp_norm;

		if (info->is_x)
		{
			for(int row_id = 0; row_id < info->disp.rows; ++row_id)
			{
				float * row_ptr = info->disp.ptr<float>(row_id);
				for(int column_id = 0; column_id < info->disp.cols; ++column_id)
				{
					float column_id_f = static_cast<float>(column_id);
					row_ptr[column_id] = row_ptr[column_id] * mult + column_id_f;
				}
			}
		}
		else
		{
			for(int row_id = 0; row_id < info->disp.rows; ++row_id)
			{
				float row_id_f = static_cast<float>(row_id);
				float * row_ptr = info->disp.ptr<float>(row_id);
				for(int column_id = 0; column_id < info->disp.cols; ++column_id)
				{
					row_ptr[column_id] = row_ptr[column_id] * mult + row_id_f;
				}
			}
		}

		{
			boost::lock_guard<boost::mutex> lock(info->is_ready_mutex);
			info->is_ready = true;
		}
		info->is_ready_condition.notify_one();
	}
}
