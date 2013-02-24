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

#include <vector>
#include <ostream>
#include <istream>
#include <memory>

namespace nnforge
{
	class data_scale_params
	{
	public:
		data_scale_params();

		data_scale_params(unsigned int feature_map_count);

		void write(std::ostream& output_stream) const;

		void read(std::istream& input_stream);

		unsigned int feature_map_count;
		std::vector<float> addition_list;
		std::vector<float> multiplication_list;
	};

	typedef std::tr1::shared_ptr<data_scale_params> data_scale_params_smart_ptr;
	typedef std::tr1::shared_ptr<const data_scale_params> const_data_scale_params_smart_ptr;
}
