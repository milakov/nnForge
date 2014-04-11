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

#include "data_transformer.h"
#include "feature_map_data_stat.h"
#include "nn_types.h"

#include <vector>
#include <ostream>
#include <istream>

#include <boost/uuid/uuid.hpp>

namespace nnforge
{
	class normalize_data_transformer : public data_transformer
	{
	public:
		normalize_data_transformer();

		normalize_data_transformer(const std::vector<feature_map_data_stat>& feature_map_data_stat_list);

		virtual ~normalize_data_transformer();

		virtual void transform(
			const void * data,
			void * data_transformed,
			neuron_data_type::input_type type,
			const layer_configuration_specific& original_config,
			unsigned int sample_id);
			
		// The stream should be created with std::ios_base::binary flag
		// The method modifies binary_stream_to_write_to to throw exceptions in case of failure
		void write(std::ostream& binary_stream_to_write_to) const;

		// The stream should be created with std::ios_base::binary flag
		// The method modifies binary_stream_to_read_from to throw exceptions in case of failure
		void read(std::istream& binary_stream_to_read_from);

		nnforge_shared_ptr<normalize_data_transformer> get_inverted_transformer() const;

	public:
		std::vector<std::pair<float, float> > mul_add_list;

	private:
		struct normalize_helper_struct
		{
			normalize_helper_struct(float mult, float add)
				: mult(mult)
				, add(add)
			{
			}

			float operator()(float x) {return x * mult + add;}

			float mult;
			float add;
		};

		static const boost::uuids::uuid normalizer_guid;
	};

	typedef nnforge_shared_ptr<normalize_data_transformer> normalize_data_transformer_smart_ptr;
}
