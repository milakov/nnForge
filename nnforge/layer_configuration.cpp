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

#include "layer_configuration.h"

namespace nnforge
{
	layer_configuration::layer_configuration()
		: feature_map_count(-1), dimension_count(-1)
	{
	}

	layer_configuration::layer_configuration(
		int feature_map_count,
		int dimension_count)
		: feature_map_count(feature_map_count), dimension_count(dimension_count)
	{
	}
}
