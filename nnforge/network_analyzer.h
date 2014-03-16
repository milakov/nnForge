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

#include "network_schema.h"
#include "network_data.h"
#include "neuron_data_type.h"
#include "layer_configuration_specific_snapshot.h"

#include <vector>
#include <utility>

namespace nnforge
{
	class network_analyzer
	{
	public:
		virtual ~network_analyzer();

		void set_data(network_data_smart_ptr data);

		void set_input_configuration_specific(const layer_configuration_specific& input_configuration_specific);

		// You need to call set_input_configuration_specific and set_data before you call this method for the 1st time
		void set_input_data(
			const void * input,
			neuron_data_type::input_type type_code,
			unsigned int input_neuron_count);

		// You need to call set_input_data before you call this method for the 1st time
		std::pair<layer_configuration_specific_snapshot_smart_ptr, layer_configuration_specific_snapshot_smart_ptr> run_backprop(
			const layer_configuration_specific_snapshot& output_data,
			const std::vector<unsigned int>& output_offset_list,
			unsigned int output_layer_id);

	protected:
		network_analyzer(network_schema_smart_ptr schema);

		// The method is called when client calls set_data. The data is guaranteed to be compatible with schema
		virtual void actual_set_data(network_data_smart_ptr data) = 0;

		// The method is called when client calls set_input_data. Input data is guaranteed to be compatible with schema
		virtual void actual_set_input_data(
			const void * input,
			neuron_data_type::input_type type_code) = 0;

		// The method is called when client calls run_backprop.
		// Output configuration (along with output_offset_list) is gueranteed to be compatible with specific configuration.
		virtual std::pair<layer_configuration_specific_snapshot_smart_ptr, layer_configuration_specific_snapshot_smart_ptr> actual_run_backprop(
			const layer_configuration_specific_snapshot& output_data,
			const std::vector<unsigned int>& output_offset_list,
			unsigned int output_layer_id,
			const std::vector<std::pair<unsigned int, unsigned int> >& input_rectangle_borders) = 0;

		// The method is called when client calls set_input_configuration_specific and the convolution specific configuration is modified.
		// The layer_config_list is guaranteed to be compatible with schema
		virtual void layer_config_list_modified() = 0;

	protected:
		network_schema_smart_ptr schema;
		layer_configuration_specific_list layer_config_list;

	private:
		network_analyzer();
		network_analyzer(const network_analyzer&);
		network_analyzer& operator =(const network_analyzer&);
	};

	typedef std::tr1::shared_ptr<network_analyzer> network_analyzer_smart_ptr;
}
