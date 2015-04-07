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

#include "layer.h"

namespace nnforge
{
	// Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
	// Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
	// http://arxiv.org/abs/1502.01852
	class parametric_rectified_linear_layer : public layer
	{
	public:
		parametric_rectified_linear_layer(unsigned int feature_map_count);

		virtual layer_smart_ptr clone() const;

		virtual float get_forward_flops(const layer_configuration_specific& input_configuration_specific) const;

		virtual float get_backward_flops(const layer_configuration_specific& input_configuration_specific) const;

		virtual float get_weights_update_flops(const layer_configuration_specific& input_configuration_specific) const;

		virtual const boost::uuids::uuid& get_uuid() const;

		virtual const std::string& get_type_name() const;

		virtual layer_configuration get_layer_configuration(const layer_configuration& input_configuration) const;

		virtual layer_configuration_specific get_output_layer_configuration_specific(const layer_configuration_specific& input_configuration_specific) const;

		virtual void write(std::ostream& binary_stream_to_write_to) const;

		virtual void write_proto(void * layer_proto) const;

		virtual void read(
			std::istream& binary_stream_to_read_from,
			const boost::uuids::uuid& layer_read_guid);

		virtual void read_proto(const void * layer_proto);

		virtual data_config get_data_config() const;

		virtual void randomize_data(
			layer_data& data,
			layer_data_custom& data_custom,
			random_generator& generator) const;

		virtual layer_data_configuration_list get_layer_data_configuration_list() const;

		virtual std::set<unsigned int> get_weight_decay_part_id_set() const;

		static const boost::uuids::uuid layer_guid;

		static const std::string layer_type_name;

	private:
		void check();

	public:
		unsigned int feature_map_count;
	};
}
