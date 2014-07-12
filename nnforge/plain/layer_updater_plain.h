/*
 *  Copyright 2011-2014 Maxim Milakov
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

#include <memory>
#include <boost/uuid/uuid.hpp>

#include "../layer.h"
#include "../nn_types.h"

#include "plain_running_configuration.h"
#include "buffer_plain_size_configuration.h"

namespace nnforge
{
	namespace plain
	{
		typedef nnforge_shared_ptr<std::vector<float> > additional_buffer_smart_ptr;
		typedef nnforge_shared_ptr<const std::vector<float> > const_additional_buffer_smart_ptr;
		struct updater_additional_buffer_set
		{
			additional_buffer_smart_ptr output_neurons_buffer;
			additional_buffer_smart_ptr input_errors_buffer;
			std::vector<additional_buffer_smart_ptr> additional_buffers;
		};

		class layer_updater_plain
		{
		public:
			virtual ~layer_updater_plain();

			virtual const boost::uuids::uuid& get_uuid() const = 0;

			void update_buffer_configuration(
				buffer_plain_size_configuration& buffer_configuration,
				const_layer_smart_ptr layer_schema,
				const layer_configuration_specific& input_configuration_specific,
				const layer_configuration_specific& output_configuration_specific,
				plain_running_configuration_const_smart_ptr plain_config,
				bool backprop_required) const;

			void update_buffer_configuration(
				buffer_plain_size_configuration& buffer_configuration,
				const_layer_smart_ptr layer_schema,
				const layer_configuration_specific& input_configuration_specific,
				const layer_configuration_specific& output_configuration_specific,
				plain_running_configuration_const_smart_ptr plain_config,
				bool backprop_required,
				unsigned int updater_entry_count) const;

			updater_additional_buffer_set allocate_additional_buffers(
				unsigned int updater_entry_count,
				const_layer_smart_ptr layer_schema,
				const layer_configuration_specific& input_configuration_specific,
				const layer_configuration_specific& output_configuration_specific,
				plain_running_configuration_const_smart_ptr plain_config,
				bool backprop_required) const;

			virtual void test(
				const_additional_buffer_smart_ptr input_buffer,
				additional_buffer_smart_ptr output_buffer,
				std::vector<additional_buffer_smart_ptr>& additional_buffers,
				plain_running_configuration_const_smart_ptr plain_config,
				const_layer_smart_ptr layer_schema,
				const_layer_data_smart_ptr data,
				const layer_configuration_specific& input_configuration_specific,
				const layer_configuration_specific& output_configuration_specific,
				unsigned int updater_count,
				unsigned int offset_input_entry_id) const = 0;

			virtual void backprop(
				additional_buffer_smart_ptr input_errors,
				const_additional_buffer_smart_ptr input_neurons,
				const_additional_buffer_smart_ptr output_errors,
				const_additional_buffer_smart_ptr output_neurons,
				std::vector<additional_buffer_smart_ptr>& additional_buffers,
				plain_running_configuration_const_smart_ptr plain_config,
				const_layer_smart_ptr layer_schema,
				const_layer_data_smart_ptr data,
				const layer_configuration_specific& input_configuration_specific,
				const layer_configuration_specific& output_configuration_specific,
				unsigned int updater_count) const = 0;

			virtual void update_weights(
				const_additional_buffer_smart_ptr input_neurons,
				const_additional_buffer_smart_ptr output_errors,
				std::vector<additional_buffer_smart_ptr>& additional_buffers,
				layer_data_smart_ptr gradient,
				plain_running_configuration_const_smart_ptr plain_config,
				const_layer_smart_ptr layer_schema,
				const layer_configuration_specific& input_configuration_specific,
				const layer_configuration_specific& output_configuration_specific,
				unsigned int updater_count,
				unsigned int offset_input_entry_id) const;

		protected:
			layer_updater_plain();

			virtual bool is_in_place_backprop() const = 0;

			virtual std::vector<std::pair<unsigned int, bool> > get_elem_count_and_per_entry_flag_additional_buffers(
				const_layer_smart_ptr layer_schema,
				const layer_configuration_specific& input_configuration_specific,
				const layer_configuration_specific& output_configuration_specific,
				plain_running_configuration_const_smart_ptr plain_config,
				bool backprop_required) const;

		private:
			layer_updater_plain(const layer_updater_plain&);
			layer_updater_plain& operator =(const layer_updater_plain&);
		};

		typedef nnforge_shared_ptr<layer_updater_plain> layer_updater_plain_smart_ptr;
		typedef nnforge_shared_ptr<const layer_updater_plain> const_layer_updater_plain_smart_ptr;
		typedef std::vector<const_layer_updater_plain_smart_ptr> const_layer_updater_plain_list;
	}
}
