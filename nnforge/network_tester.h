/*
 *  Copyright 2011-2015 Maxim Milakov
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
#include "supervised_data_reader.h"
#include "unsupervised_data_reader.h"
#include "testing_complete_result_set.h"
#include "layer_configuration_specific.h"
#include "layer_configuration_specific_snapshot.h"
#include "neuron_data_type.h"
#include "nn_types.h"

#include <vector>
#include <utility>

namespace nnforge
{
	class network_tester
	{
	public:
		virtual ~network_tester();

		void set_data(network_data_smart_ptr data);

		void clear_data();

		// You don't need to call this method before calling test with supervised_data_reader
		void set_input_configuration_specific(const layer_configuration_specific& input_configuration_specific);

		void test(
			supervised_data_reader& reader,
			testing_complete_result_set& result);

		output_neuron_value_set_smart_ptr run(
			unsupervised_data_reader& reader,
			unsigned int sample_count);

		// You need to call set_input_configuration_specific before you call this method for the 1st time
		std::vector<layer_configuration_specific_snapshot_smart_ptr> get_snapshot(
			const void * input,
			neuron_data_type::input_type type_code,
			unsigned int input_neuron_count);

		// You need to call set_input_configuration_specific before you call this method for the 1st time
		std::vector<layer_configuration_specific_snapshot_smart_ptr> get_snapshot(const std::vector<unsigned char>& input);

		// You need to call set_input_configuration_specific before you call this method for the 1st time
		std::vector<layer_configuration_specific_snapshot_smart_ptr> get_snapshot(const std::vector<float>& input);

		// You need to call set_input_configuration_specific before you call this method for the 1st time
		layer_configuration_specific_snapshot_smart_ptr run(
			const void * input,
			neuron_data_type::input_type type_code,
			unsigned int input_neuron_count);

		// You need to call set_input_configuration_specific before you call this method for the 1st time
		layer_configuration_specific_snapshot_smart_ptr run(const std::vector<unsigned char>& input);

		// You need to call set_input_configuration_specific before you call this method for the 1st time
		layer_configuration_specific_snapshot_smart_ptr run(const std::vector<float>& input);

		// set_input_configuration_specific should be called prior to this method call for this method to succeed
		float get_flops_for_single_entry() const;

		const_network_schema_smart_ptr get_schema() const;

	protected:
		network_tester(network_schema_smart_ptr schema);

		// schema, data and reader are guaranteed to be compatible
		virtual output_neuron_value_set_smart_ptr actual_run(
			unsupervised_data_reader& reader,
			unsigned int sample_count) = 0;

		// The method is called when client calls set_data. The data is guaranteed to be compatible with schema
		virtual void actual_set_data(network_data_smart_ptr data) = 0;

		virtual void actual_clear_data() = 0;

		// The method is called when client calls get_snapshot. The data is guaranteed to be compatible with schema
		virtual std::vector<layer_configuration_specific_snapshot_smart_ptr> actual_get_snapshot(
			const void * input,
			neuron_data_type::input_type type_code) = 0;

		// The method is called when client calls get_snapshot. The data is guaranteed to be compatible with schema
		virtual layer_configuration_specific_snapshot_smart_ptr actual_run(
			const void * input,
			neuron_data_type::input_type type_code) = 0;

		// The method is called when client calls set_input_configuration_specific and the convolution specific configuration is modified.
		// The layer_config_list is guaranteed to be compatible with schema
		virtual void layer_config_list_modified() = 0;

		void update_flops();

	protected:
		network_schema_smart_ptr schema;
		layer_configuration_specific_list layer_config_list;
		std::vector<unsigned int> cumulative_tiling_factor_list;
		float flops;

	private:
		network_tester();
		network_tester(const network_tester&);
		network_tester& operator =(const network_tester&);
	};

	typedef nnforge_shared_ptr<network_tester> network_tester_smart_ptr;
}
