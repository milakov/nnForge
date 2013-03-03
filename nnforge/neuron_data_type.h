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

#include <memory>

namespace nnforge
{
	class neuron_data_type
	{
	public:
		enum input_type
		{
			type_unknown = 0,
			type_byte = 1,
			type_float = 2
		};

		static size_t get_input_size(input_type t);

	private:
		neuron_data_type();
		neuron_data_type(const neuron_data_type&);
		neuron_data_type& operator =(const neuron_data_type&);
	};
}
