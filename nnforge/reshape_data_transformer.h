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

#include "data_transformer.h"

namespace nnforge
{
	class reshape_data_transformer : public data_transformer
	{
	public:
		reshape_data_transformer(const layer_configuration_specific& config);

		~reshape_data_transformer();

		virtual void transform(
			const void * data,
			void * data_transformed,
			neuron_data_type::input_type type,
			const layer_configuration_specific& original_config,
			unsigned int sample_id);

		virtual layer_configuration_specific get_transformed_configuration(const layer_configuration_specific& original_config) const;

		virtual bool is_in_place() const;

		virtual bool is_deterministic() const;

	protected:
		layer_configuration_specific config;
	};
}
