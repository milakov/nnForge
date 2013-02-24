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

#include "network_schema.h"
#include "network_data.h"
#include "layer_configuration_specific.h"
#include "supervised_data_reader.h"
#include "data_scale_params.h"

#include <memory>

namespace nnforge
{
	class hessian_calculator
	{
	public:
		virtual ~hessian_calculator();

		// You don't need to call this method before calling get_hessian with supervised_data_reader_byte
		void set_input_configuration_specific(const layer_configuration_specific& input_configuration_specific);

		network_data_smart_ptr get_hessian(
			supervised_data_reader_byte& reader,
			network_data_smart_ptr data,
			unsigned int hessian_entry_to_process_count);

		// set_input_configuration_specific should be called prior to this method call for this method to succeed
		float get_flops_for_single_entry() const;

	protected:
		hessian_calculator(
			network_schema_smart_ptr schema,
			const_data_scale_params_smart_ptr scale_params);

		// schema, data and reader are guaranteed to be compatible
		virtual network_data_smart_ptr actual_get_hessian(
			supervised_data_reader_byte& reader,
			network_data_smart_ptr data,
			unsigned int hessian_entry_to_process_count) = 0;

		// The method is called when client calls set_input_configuration_specific and the convolution specific configuration is modified.
		// The layer_config_list is guaranteed to be compatible with schema
		virtual void layer_config_list_modified() = 0;

		void update_flops();

	protected:
		network_schema_smart_ptr schema;
		layer_configuration_specific_list layer_config_list;
		const_data_scale_params_smart_ptr current_scale_params; // Defined in set_input_configuration_specific
		float flops;

	private:
		hessian_calculator();
		hessian_calculator(const hessian_calculator&);
		hessian_calculator& operator =(const hessian_calculator&);

		const_data_scale_params_smart_ptr scale_params;
	};

	typedef std::tr1::shared_ptr<hessian_calculator> hessian_calculator_smart_ptr;
}
