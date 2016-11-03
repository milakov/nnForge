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

#include <memory>

namespace nnforge
{
	class data_transformer
	{
	public:
		typedef std::shared_ptr<data_transformer> ptr;

		virtual ~data_transformer() = default;

		virtual void transform(
			const float * data,
			float * data_transformed,
			const layer_configuration_specific& original_config,
			unsigned int sample_id) = 0;

		virtual layer_configuration_specific get_transformed_configuration(const layer_configuration_specific& original_config) const;

		virtual unsigned int get_sample_count() const;

	protected:
		data_transformer() = default;

	private:
		data_transformer(const data_transformer&) = delete;
		data_transformer& operator =(const data_transformer&) = delete;
	};
}
