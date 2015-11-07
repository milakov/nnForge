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

#include "noise_data_transformer.h"

#include "neural_network_exception.h"

#include <opencv2/core/core.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	noise_data_transformer::noise_data_transformer(float max_noise)
	{
		generator = rnd::get_random_generator();

		max_noise_distribution = nnforge_uniform_real_distribution<float>(-max_noise, max_noise);
	}

	noise_data_transformer::~noise_data_transformer()
	{
	}

	void noise_data_transformer::transform(
		const float * data,
		float * data_transformed,
		const layer_configuration_specific& original_config,
		unsigned int sample_id)
	{
		unsigned int elem_count = original_config.get_neuron_count();

		{
			boost::lock_guard<boost::mutex> lock(gen_stream_mutex);

			for(unsigned int elem_id = 0; elem_id < elem_count; ++elem_id)
			{
				float shift = max_noise_distribution.min();
				if (max_noise_distribution.max() > max_noise_distribution.min())
					shift = max_noise_distribution(generator);
				data_transformed[elem_id] = data[elem_id] + shift;
			}
		}
	}
}
