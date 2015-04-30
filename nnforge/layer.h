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

#include "layer_configuration.h"
#include "layer_configuration_specific.h"
#include "layer_data.h"
#include "layer_data_custom.h"
#include "rnd.h"
#include "layer_data_configuration.h"
#include "nn_types.h"
#include "tiling_factor.h"

#include <boost/uuid/uuid.hpp>
#include <ostream>
#include <istream>
#include <set>

namespace nnforge
{
	typedef std::vector<unsigned int> data_config;
	typedef std::vector<unsigned int> data_custom_config;

	class layer
	{
	public:
		virtual ~layer();

		virtual nnforge_shared_ptr<layer> clone() const = 0;

		virtual layer_configuration get_layer_configuration(const layer_configuration& input_configuration) const;

		virtual layer_configuration_specific get_output_layer_configuration_specific(const layer_configuration_specific& input_configuration_specific) const;

		virtual layer_configuration_specific get_input_layer_configuration_specific(const layer_configuration_specific& output_configuration_specific) const;

		// Returns minimal input rectangle which this layer quasi-transforms into output one covering the one supplied as an argument to the function
		// "Quasi" means that we don't take into account "soft" effects from nearby neurons, for example when doing local contrast subtracting blurred version
		virtual std::vector<std::pair<unsigned int, unsigned int> > get_input_rectangle_borders(const std::vector<std::pair<unsigned int, unsigned int> >& output_rectangle_borders) const;

		virtual layer_data_configuration_list get_layer_data_configuration_list() const;

		virtual float get_forward_flops(const layer_configuration_specific& input_configuration_specific) const = 0;

		virtual float get_backward_flops(const layer_configuration_specific& input_configuration_specific) const = 0;

		virtual float get_weights_update_flops(const layer_configuration_specific& input_configuration_specific) const;

		virtual const boost::uuids::uuid& get_uuid() const = 0;

		virtual const std::string& get_type_name() const = 0;

		// The method shouldn't write uuid of the layer type
		// The stream should be created with std::ios_base::binary flag
		virtual void write(std::ostream& binary_stream_to_write_to) const;

		// The method shouldn't read uuid of the layer type
		// The stream should be created with std::ios_base::binary flag
		virtual void read(
			std::istream& binary_stream_to_read_from,
			const boost::uuids::uuid& layer_read_guid);

		virtual void read_proto(const void * layer_proto);

		virtual void write_proto(void * layer_proto) const;

		// All values are set to 0.0F
		layer_data_smart_ptr create_layer_data() const;

		// All values are set to -1
		layer_data_custom_smart_ptr create_layer_data_custom() const;

		// The method throws exception in case the data is not suitable for the layer
		void check_layer_data_consistency(const layer_data& data) const;

		// The method throws exception in case the data is not suitable for the layer
		void check_layer_data_custom_consistency(const layer_data_custom& data_custom) const;

		// Override this member function to randomize data
		virtual void randomize_data(
			layer_data& data,
			layer_data_custom& data_custom,
			random_generator& generator) const;

		// Override this member function to randomize data
		virtual void randomize_orthogonal_data(
			layer_data& data,
			layer_data_custom& data_custom,
			random_generator& generator) const;

		virtual std::set<unsigned int> get_weight_decay_part_id_set() const;

		bool is_empty_data() const;

		bool is_empty_data_custom() const;

		virtual std::vector<tiling_factor> get_tiling_factor_list() const;

		tiling_factor get_tiling_factor() const;

	public:
		std::string instance_name;

	protected:
		layer();

		virtual data_config get_data_config() const;

		virtual data_custom_config get_data_custom_config() const;
	};

	typedef nnforge_shared_ptr<layer> layer_smart_ptr;
	typedef nnforge_shared_ptr<const layer> const_layer_smart_ptr;
	typedef std::vector<const_layer_smart_ptr> const_layer_list;
}
