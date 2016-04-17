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

#include "layer.h"

#include <vector>
#include <limits>

namespace nnforge
{
	// The layer contains 4 chunks of weights: gamma, beta, mean, inverse sigma
	class batch_norm_layer : public layer
	{
	public:
		batch_norm_layer(
			unsigned int feature_map_count,
			float epsilon = default_batch_normalization_epsilon);

		virtual layer::ptr clone() const;

		virtual float get_flops_per_entry(
			const std::vector<layer_configuration_specific>& input_configuration_specific_list,
			const layer_action& action) const;

		virtual std::string get_type_name() const;

		virtual void write_proto(void * layer_proto) const;

		virtual void read_proto(const void * layer_proto);

		virtual data_config get_data_config() const;

		virtual void randomize_data(
			layer_data::ptr data,
			layer_data_custom::ptr data_custom,
			random_generator& generator) const;

		virtual layer_data_configuration_list get_layer_data_configuration_list() const;

		virtual std::set<unsigned int> get_weight_decay_part_id_set() const;

		virtual std::vector<std::string> get_parameter_strings() const;

		virtual bool has_fused_backward_data_and_weights() const;

		static const std::string layer_type_name;

	private:
		void check();

	public:
		unsigned int feature_map_count;
		float epsilon;

		static const float default_batch_normalization_epsilon;
	};
}
