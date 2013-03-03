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

#include "layer_hessian_plain.h"

#include "../hessian_calculator.h"
#include "plain_running_configuration.h"
#include "buffer_plain_size_configuration.h"
#include "layer_tester_plain.h"

namespace nnforge
{
	namespace plain
	{
		class hessian_calculator_plain : public hessian_calculator
		{
		public:
			hessian_calculator_plain(
				network_schema_smart_ptr schema,
				plain_running_configuration_const_smart_ptr plain_config);

			~hessian_calculator_plain();

		protected:
			// schema, data and reader are guaranteed to be compatible
			virtual network_data_smart_ptr actual_get_hessian(
				supervised_data_reader& reader,
				network_data_smart_ptr data,
				unsigned int hessian_entry_to_process_count);

			// The method is called when client calls set_input_configuration_specific and the convolution specific configuration is modified.
			// The layer_config_list is guaranteed to be compatible with schema
			virtual void layer_config_list_modified();

		private:
			hessian_calculator_plain(const hessian_calculator_plain&);
			hessian_calculator_plain& operator =(const hessian_calculator_plain&);

			void update_buffers_configuration(buffer_plain_size_configuration& buffer_configuration) const;

			plain_running_configuration_const_smart_ptr plain_config;

			unsigned int testing_layer_count;
			const_layer_list::const_iterator start_layer_nonempty_weights_iterator;

			const_layer_tester_plain_list tester_list;
			const_layer_hessian_plain_list hessian_list;
		};
	}
}
