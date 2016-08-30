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

#pragma once

#include "data_transformer.h"
#include "rnd.h"
#include "nn_types.h"

#include <vector>
#include <boost/thread/thread.hpp>

namespace nnforge
{
	class uniform_intensity_data_transformer : public data_transformer
	{
	public:
		uniform_intensity_data_transformer(
			const std::vector<float>& min_shift_list,
			const std::vector<float>& max_shift_list);

		virtual ~uniform_intensity_data_transformer();

		virtual void transform(
			const float * data,
			float * data_transformed,
			const layer_configuration_specific& original_config,
			unsigned int sample_id);
			
	protected:
		random_generator generator;
		boost::mutex gen_stream_mutex;

		std::vector<bool> apply_shift_distribution_list;
		std::vector<nnforge_uniform_real_distribution<float> > shift_distribution_list;
	};
}
