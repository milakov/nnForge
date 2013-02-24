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

#include "absolute_layer_hessian_schema.h"

#include "../absolute_layer.h"
#include "absolute_layer_hessian_cuda.h"

namespace nnforge
{
	namespace cuda
	{
		absolute_layer_hessian_schema::absolute_layer_hessian_schema()
		{
		}

		absolute_layer_hessian_schema::~absolute_layer_hessian_schema()
		{
		}

		std::tr1::shared_ptr<layer_hessian_schema> absolute_layer_hessian_schema::create_specific() const
		{
			return layer_hessian_schema_smart_ptr(new absolute_layer_hessian_schema());
		}

		const boost::uuids::uuid& absolute_layer_hessian_schema::get_uuid() const
		{
			return absolute_layer::layer_guid;
		}

		layer_hessian_cuda_smart_ptr absolute_layer_hessian_schema::create_hessian_specific(
			const layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific) const
		{
			return layer_hessian_cuda_smart_ptr(new absolute_layer_hessian_cuda());
		}
	}
}
