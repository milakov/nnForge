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

#include "layer_configuration_specific.h"

#include <vector>
#include <memory>

namespace nnforge
{
	class layer_configuration_specific_snapshot
	{
	public:
		typedef std::shared_ptr<layer_configuration_specific_snapshot> ptr;

		layer_configuration_specific_snapshot() = default;

		layer_configuration_specific_snapshot(const layer_configuration_specific& config);

		layer_configuration_specific config;
		std::vector<float> data;
	};
}
