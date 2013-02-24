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

#include "layer_updater_schema.h"

#include <memory>

namespace nnforge
{
	namespace cuda
	{
		class hyperbolic_tangent_layer_updater_schema : public layer_updater_schema
		{
		public:
			hyperbolic_tangent_layer_updater_schema();

			virtual ~hyperbolic_tangent_layer_updater_schema();

			virtual const boost::uuids::uuid& get_uuid() const;

		protected:
			virtual std::tr1::shared_ptr<layer_updater_schema> create_specific() const;

			virtual layer_updater_cuda_smart_ptr create_updater_specific(
				const layer_configuration_specific& input_configuration_specific,
				const layer_configuration_specific& output_configuration_specific) const;
		};
	}
}
