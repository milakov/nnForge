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

#include "plain_running_configuration.h"

#include "../layer.h"
#include "../weight_vector_bound.h"
#include "../nn_types.h"

#include <map>

namespace nnforge
{
	namespace plain
	{
		class weight_vector_bound_plain
		{
		public:
			virtual ~weight_vector_bound_plain();

			virtual const boost::uuids::uuid& get_uuid() const = 0;

			virtual void normalize_weights(
				const weight_vector_bound& bound,
				layer_data_list& data,
				plain_running_configuration_const_smart_ptr plain_config,
				const_layer_smart_ptr layer_schema,
				unsigned int updater_count) const = 0;

		protected:
			weight_vector_bound_plain();

		private:
			weight_vector_bound_plain(const weight_vector_bound_plain&);
			weight_vector_bound_plain& operator =(const weight_vector_bound_plain&);
		};

		typedef nnforge_shared_ptr<weight_vector_bound_plain> weight_vector_bound_plain_smart_ptr;
		typedef nnforge_shared_ptr<const weight_vector_bound_plain> const_weight_vector_bound_plain_smart_ptr;
		typedef std::map<unsigned int, const_weight_vector_bound_plain_smart_ptr> weight_vector_bound_map;
	}
}
