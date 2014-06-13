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

#include "layer_updater_plain.h"

#include "../network_updater.h"
#include "plain_running_configuration.h"
#include "buffer_plain_size_configuration.h"
#include "layer_tester_plain.h"
#include "weight_vector_bound_plain.h"

namespace nnforge
{
	namespace plain
	{
		class network_updater_plain : public network_updater
		{
		public:
			network_updater_plain(
				network_schema_smart_ptr schema,
				const_error_function_smart_ptr ef,
				const std::map<unsigned int, float>& layer_to_dropout_rate_map,
				const std::map<unsigned int, weight_vector_bound>& layer_to_weight_vector_bound_map,
				float weight_decay,
				plain_running_configuration_const_smart_ptr plain_config);

			~network_updater_plain();

			virtual unsigned int get_max_batch_size() const;

		protected:
			// schema, data and reader are guaranteed to be compatible
			virtual std::vector<testing_result_smart_ptr> actual_update(
				supervised_data_reader& reader,
				const std::vector<network_data_smart_ptr>& learning_rate_vector_list,
				std::vector<network_data_smart_ptr>& data_list);

			// The method is called when client calls set_input_configuration_specific and the convolution specific configuration is modified.
			// The layer_config_list is guaranteed to be compatible with schema
			virtual void layer_config_list_modified();

		private:
			network_updater_plain(const network_updater_plain&);
			network_updater_plain& operator =(const network_updater_plain&);

			void update_buffers_configuration(
				buffer_plain_size_configuration& buffer_configuration,
				unsigned int updater_entry_count) const;

			void apply_dropout(
				additional_buffer_smart_ptr target_buffer,
				const float dropout_rate,
				const unsigned int mask,
				const unsigned int updater_count,
				const unsigned int offset_in_random_list) const;

			plain_running_configuration_const_smart_ptr plain_config;

			unsigned int testing_layer_count;
			const_layer_list::const_iterator start_layer_nonempty_weights_iterator;

			const_layer_tester_plain_list tester_list;
			const_layer_updater_plain_list updater_list;
			weight_vector_bound_map weight_vector_bounds;

			static unsigned int max_entry_count_in_single_batch;
		};
	}
}
