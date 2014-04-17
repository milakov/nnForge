/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include "../neural_network_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	namespace cuda
	{
		class packed_config_util
		{
		public:
			static size_t get_packed_config_size(int dimension_size)
			{
				return ((dimension_size + 1) >> 1) * 4;
			}
		};

		template <int dimension_count>
		class __align__(4) packed_config
		{
		public:
			__host__ __device__ __forceinline__
			int get_val(int index) const
			{
				unsigned int val = raw_data[index >> 1];
				if (index & 1)
					val >>= 16;

				return (int)(val & 0xFFFF);
			}

			void set_val(int index, int val)
			{
				if (val > 0xFFFF)
					throw neural_network_exception((boost::format("Too large value to be save into packed config: %1%") % val).str());

				unsigned int new_val = (unsigned int)val;
				unsigned int mask = 0xFFFF;
				if (index & 1)
				{
					new_val <<= 16;
					mask <<= 16;
				}

				raw_data[index >> 1] = (raw_data[index >> 1] & ~mask) | new_val;
			}

		private:
			static const int int_elem_count = (dimension_count + 1) >> 1;
			unsigned int raw_data[int_elem_count];
		};
	}
}
