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
			typedef std::shared_ptr<layer_testing_schema> ptr;
			typedef std::shared_ptr<const layer_testing_schema> const_ptr;

			virtual ~layer_testing_schema() = default;

			std::shared_ptr<layer_testing_schema> create(layer::const_ptr layer_schema) const;

			layer_tester_cuda::ptr create_tester(
				const std::vector<layer_configuration_specific>& input_configuration_specific_list,
				const layer_configuration_specific& output_configuration_specific,
				cuda_running_configuration::const_ptr cuda_config) const;

			virtual std::string get_type_name() const = 0;

			// returns the list of buffers defining the schema
			virtual std::vector<cuda_linear_buffer_device::const_ptr> get_schema_buffers() const;

		protected:
			virtual std::shared_ptr<layer_testing_schema> create_specific() const = 0;

			virtual layer_tester_cuda::ptr create_tester_specific(
				const std::vector<layer_configuration_specific>& input_configuration_specific_list,
				const layer_configuration_specific& output_configuration_specific,
				const cuda_running_configuration& cuda_config) const = 0;

			layer_testing_schema() = default;

			layer::const_ptr layer_schema;

		private:
			layer_testing_schema(const layer_testing_schema&) = delete;
			layer_testing_schema& operator =(const layer_testing_schema&) = delete;
		};
	}
}
