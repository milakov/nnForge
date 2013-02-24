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

#include "../layer.h"
#include "../layer_data.h"
#include "../layer_configuration_specific.h"
#include "layer_tester_cuda.h"
#include "cuda_running_configuration.h"

#include <utility>
#include <memory>
#include <vector>
#include <string>
#include <boost/uuid/uuid.hpp>

namespace nnforge
{
	namespace cuda
	{
		class layer_testing_schema
		{
		public:
			virtual ~layer_testing_schema();

			std::tr1::shared_ptr<layer_testing_schema> create(
				const_layer_smart_ptr layer_schema,
				cuda_running_configuration_const_smart_ptr cuda_config) const;

			layer_tester_cuda_smart_ptr create_tester(
				const layer_configuration_specific& input_configuration_specific,
				const layer_configuration_specific& output_configuration_specific) const;

			virtual const boost::uuids::uuid& get_uuid() const = 0;

			// returns the list of buffers defining the schema
			virtual std::vector<const_cuda_linear_buffer_device_smart_ptr> get_schema_buffers() const;

		protected:
			virtual std::tr1::shared_ptr<layer_testing_schema> create_specific() const = 0;

			virtual layer_tester_cuda_smart_ptr create_tester_specific(
				const layer_configuration_specific& input_configuration_specific,
				const layer_configuration_specific& output_configuration_specific) const = 0;

			layer_testing_schema();

			const_layer_smart_ptr layer_schema;
			cuda_running_configuration_const_smart_ptr cuda_config;

		private:
			layer_testing_schema(const layer_testing_schema&);
			layer_testing_schema& operator =(const layer_testing_schema&);
		};

		typedef std::tr1::shared_ptr<layer_testing_schema> layer_testing_schema_smart_ptr;
		typedef std::tr1::shared_ptr<const layer_testing_schema> const_layer_testing_schema_smart_ptr;
		typedef std::vector<const_layer_testing_schema_smart_ptr> const_layer_testing_schema_list;
	}
}
