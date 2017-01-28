/*
 *  Copyright 2011-2017 Maxim Milakov
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

#include "layer_configuration_specific.h"
#include "layer_data.h"
#include "layer_data_custom.h"
#include "rnd.h"
#include "layer_data_configuration.h"
#include "tiling_factor.h"
#include "layer_action.h"

#include <ostream>
#include <istream>
#include <set>
#include <memory>

namespace nnforge
{
	typedef std::vector<unsigned int> data_config;
	typedef std::vector<unsigned int> data_custom_config;

	class layer
	{
	public:
		typedef std::shared_ptr<layer> ptr;
		typedef std::shared_ptr<const layer> const_ptr;

		virtual ~layer() = default;

		virtual layer::ptr clone() const = 0;

		virtual layer_configuration_specific get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const;

		// The method returns false in case layer is unable to set input_configuration_specific
		virtual bool get_input_layer_configuration_specific(
			layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int input_layer_id) const;

		virtual layer_data_configuration_list get_layer_data_configuration_list() const;

		// return flops per output entry
		virtual float get_flops_per_entry(
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_action& action) const = 0;

		virtual std::string get_type_name() const = 0;

		virtual void read_proto(const void * layer_proto);

		virtual void write_proto(void * layer_proto) const;

		// All values are set to 0.0F
		layer_data::ptr create_layer_data() const;

		// All values are set to -1
		layer_data_custom::ptr create_layer_data_custom() const;

		// The method throws exception in case the data is not suitable for the layer
		void check_layer_data_consistency(const layer_data& data) const;

		// The method throws exception in case the data is not suitable for the layer
		void check_layer_data_custom_consistency(const layer_data_custom& data_custom) const;

		// Override this member function to randomize data
		virtual void randomize_data(
			layer_data::ptr data,
			layer_data_custom::ptr data_custom,
			random_generator& generator) const;

		// Override this member function to randomize data
		virtual void randomize_orthogonal_data(
			layer_data::ptr data,
			layer_data_custom::ptr data_custom,
			random_generator& generator) const;

		virtual std::set<unsigned int> get_weight_decay_part_id_set() const;

		bool is_empty_data() const;

		bool is_empty_data_custom() const;

		virtual tiling_factor get_tiling_factor() const;

		virtual std::string get_string_for_average_data(
			const layer_configuration_specific& config,
			const std::vector<double>& data) const;

		virtual std::vector<std::string> get_parameter_strings() const;

		virtual bool has_fused_backward_data_and_weights() const;

		// Return true in case backwad data actions is just identity, that is gradient is 1
		virtual bool is_backward_data_identity(int backprop_index) const;

	public:
		std::string instance_name;
		std::vector<std::string> input_layer_instance_names;

	protected:
		layer() = default;

		virtual data_config get_data_config() const;

		virtual data_custom_config get_data_custom_config() const;
	};
}
