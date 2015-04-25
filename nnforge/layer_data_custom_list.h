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

#include "layer_data_custom.h"
#include "layer.h"

#include <vector>

namespace nnforge
{
	class layer_data_custom_list : public std::vector<layer_data_custom_smart_ptr>
	{
	public:
		layer_data_custom_list();

		layer_data_custom_list(const const_layer_list& layer_list);

		void check_consistency(const const_layer_list& layer_list) const;
	};

	typedef nnforge_shared_ptr<layer_data_custom_list> layer_data_custom_list_smart_ptr;
	typedef nnforge_shared_ptr<const layer_data_custom_list> const_layer_data_custom_list_smart_ptr;
}
