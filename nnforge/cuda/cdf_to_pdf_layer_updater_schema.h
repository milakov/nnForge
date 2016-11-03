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

#include "layer_updater_schema.h"

namespace nnforge
{
	namespace cuda
	{
		class cdf_to_pdf_layer_updater_schema : public layer_updater_schema
		{
		public:
			cdf_to_pdf_layer_updater_schema() = default;

			virtual ~cdf_to_pdf_layer_updater_schema() = default;

			virtual std::string get_type_name() const;

		protected:
			virtual layer_updater_schema::ptr create_specific() const;

			virtual layer_updater_cuda::ptr create_updater_specific(
				const std::vector<layer_configuration_specific>& input_configuration_specific_list,
				const layer_configuration_specific& output_configuration_specific) const;
		};
	}
}
