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
#include "feature_map_data_stat.h"

#include <memory>
#include <vector>

namespace nnforge
{
	class normalize_data_transformer : public data_transformer
	{
	public:
		typedef std::shared_ptr<normalize_data_transformer> ptr;

		normalize_data_transformer() = default;

		normalize_data_transformer(const std::vector<feature_map_data_stat>& feature_map_data_stat_list);

		virtual ~normalize_data_transformer() = default;

		virtual void transform(
			const float * data,
			float * data_transformed,
			const layer_configuration_specific& original_config,
			unsigned int sample_id);
			
		void write_proto(std::ostream& stream_to_write_to) const;

		void read_proto(std::istream& stream_to_read_from);

		normalize_data_transformer::ptr get_inverted_transformer() const;

	public:
		std::vector<std::pair<float, float> > mul_add_list;
	};
}
