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

#include "data_transformer.h"

namespace nnforge
{
	data_transformer::data_transformer()
	{
	}

	data_transformer::~data_transformer()
	{
	}

	layer_configuration_specific data_transformer::get_transformed_configuration(const layer_configuration_specific& original_config) const
	{
		return original_config;
	}

	bool data_transformer::is_in_place() const
	{
		return true;
	}

	void data_transformer::reset()
	{
	}

	unsigned int data_transformer::get_sample_count() const
	{
		return 1;
	}
}
