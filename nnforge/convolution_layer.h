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

#include "layer.h"

#include <vector>

namespace nnforge
{
	class convolution_layer : public layer
	{
	public:
		convolution_layer(
			const std::vector<unsigned int>& window_sizes,
			unsigned int input_feature_map_count,
			unsigned int output_feature_map_count,
			const std::vector<unsigned int>& left_zero_padding = std::vector<unsigned int>(),
			const std::vector<unsigned int>& right_zero_padding = std::vector<unsigned int>(),
			const std::vector<unsigned int>& strides = std::vector<unsigned int>(),
			bool bias = true,
			const std::vector<unsigned int>& dilation = std::vector<unsigned int>());

		virtual layer::ptr clone() const;

		virtual layer_configuration_specific get_output_layer_configuration_specific(const std::vector<layer_configuration_specific>& input_configuration_specific_list) const;

		virtual bool get_input_layer_configuration_specific(
			layer_configuration_specific& input_configuration_specific,
			const layer_configuration_specific& output_configuration_specific,
			unsigned int input_layer_id) const;

		virtual layer_data_configuration_list get_layer_data_configuration_list() const;

		virtual float get_flops_per_entry(
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_action& action) const;

		virtual std::string get_type_name() const;

		virtual void write_proto(void * layer_proto) const;

		virtual void read_proto(const void * layer_proto);

		virtual void randomize_data(
			layer_data::ptr data,
			layer_data_custom::ptr data_custom,
			random_generator& generator) const;

		virtual void randomize_orthogonal_data(
			layer_data::ptr data,
			layer_data_custom::ptr data_custom,
			random_generator& generator) const;

		virtual std::set<unsigned int> get_weight_decay_part_id_set() const;

		virtual std::vector<std::string> get_parameter_strings() const;

		static const std::string layer_type_name;

	protected:
		virtual data_config get_data_config() const;

	private:
		void check();

	public:
		std::vector<unsigned int> window_sizes;
		unsigned int input_feature_map_count;
		unsigned int output_feature_map_count;
		std::vector<unsigned int> left_zero_padding;
		std::vector<unsigned int> right_zero_padding;
		std::vector<unsigned int> strides;
		std::vector<unsigned int> dilation;
		bool bias;
	};
}
