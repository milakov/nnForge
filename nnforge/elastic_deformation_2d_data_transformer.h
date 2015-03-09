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

#include "data_transformer.h"
#include "rnd.h"
#include "nn_types.h"

#include <opencv2/core/core.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>

#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501
#endif 

#include <boost/asio/io_service.hpp>

namespace nnforge
{
	class elastic_deformation_2d_data_transformer : public data_transformer
	{
	public:
		elastic_deformation_2d_data_transformer(
			float sigma, // >0
			float alpha,
			unsigned char border_value);

		virtual ~elastic_deformation_2d_data_transformer();

		virtual void transform(
			const void * data,
			void * data_transformed,
			neuron_data_type::input_type type,
			const layer_configuration_specific& original_config,
			unsigned int sample_id);
			
		virtual bool is_deterministic() const;

	private:
		class smooth_info
		{
		public:
			smooth_info();

			cv::Mat1f disp;
			int ksize;
			float sigma;
			float alpha;
			bool is_x;
			bool is_ready;
			boost::mutex is_ready_mutex;
			boost::condition_variable is_ready_condition;
		};

	protected:
		float alpha;
		float sigma;
		unsigned char border_value;

		random_generator gen;
		nnforge_uniform_real_distribution<float> displacement_distribution;

	private:
		boost::asio::io_service io_service;
		boost::thread_group threadpool;
		boost::asio::io_service::work work;

		static void smooth_worker(nnforge_shared_ptr<smooth_info> info);
	};
}
