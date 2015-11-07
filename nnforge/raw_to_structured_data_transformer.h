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

#include "nn_types.h"
#include "layer_configuration_specific.h"
#include <vector>

namespace nnforge
{
	class raw_to_structured_data_transformer
	{
	public:
		typedef nnforge_shared_ptr<raw_to_structured_data_transformer> ptr;

		virtual ~raw_to_structured_data_transformer();

		virtual void transform(
			unsigned int sample_id,
			const std::vector<unsigned char>& raw_data,
			float * structured_data) = 0;

		virtual layer_configuration_specific get_configuration() const = 0;

		virtual unsigned int get_sample_count() const;

	protected:
		raw_to_structured_data_transformer();

	private:
		raw_to_structured_data_transformer(const raw_to_structured_data_transformer&);
		raw_to_structured_data_transformer& operator =(const raw_to_structured_data_transformer&);
	};
}
