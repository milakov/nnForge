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

#include "layer_configuration.h"
#include "layer_configuration_specific.h"
#include "layer_data.h"
#include "rnd.h"
#include "dropout_layer_config.h"
#include "layer_data_configuration.h"
#include "nn_types.h"

#include <boost/uuid/uuid.hpp>
#include <ostream>
#include <istream>

namespace nnforge
{
	typedef std::vector<unsigned int> data_config;

	class layer
	{
	public:
		virtual ~layer();

		virtual nnforge_shared_ptr<layer> clone() const = 0;

		virtual layer_configuration get_layer_configuration(const layer_configuration& input_configuration) const;

		virtual layer_configuration_specific get_output_layer_configuration_specific(const layer_configuration_specific& input_configuration_specific) const;

		// Returns minimal input rectangle which this layer quasi-transforms into output one covering the one supplied as an argument to the function
		// "Quasi" means that we don't take into account "soft" effects from nearby neurons, for example when doing local contrast subtracting blurred version
		virtual std::vector<std::pair<unsigned int, unsigned int> > get_input_rectangle_borders(const std::vector<std::pair<unsigned int, unsigned int> >& output_rectangle_borders) const;

		virtual layer_data_configuration_list get_layer_data_configuration_list() const;

		virtual float get_forward_flops(const layer_configuration_specific& input_configuration_specific) const = 0;

		virtual float get_backward_flops(const layer_configuration_specific& input_configuration_specific) const = 0;

		virtual float get_weights_update_flops(const layer_configuration_specific& input_configuration_specific) const;

		virtual float get_backward_flops_2nd(const layer_configuration_specific& input_configuration_specific) const = 0;

		virtual float get_weights_update_flops_2nd(const layer_configuration_specific& input_configuration_specific) const;

		virtual const boost::uuids::uuid& get_uuid() const = 0;

		// The method shouldn't write uuid of the layer type
		// The stream should be created with std::ios_base::binary flag
		virtual void write(std::ostream& binary_stream_to_write_to) const;

		// The method shouldn't read uuid of the layer type
		// The stream should be created with std::ios_base::binary flag
		virtual void read(std::istream& binary_stream_to_read_from);

		// All values are set to 0.0F
		layer_data_smart_ptr create_layer_data() const;

		// The method throws exception in case the data is not suitable for the layer
		void check_layer_data_consistency(const layer_data& data) const;

		// Override this member function to randomize data
		virtual void randomize_data(
			layer_data& data,
			random_generator& generator) const;

		virtual dropout_layer_config get_dropout_layer_config(float dropout_rate) const;

		bool is_empty_data() const;

	protected:
		layer();

		virtual data_config get_data_config() const;
	};

	typedef nnforge_shared_ptr<layer> layer_smart_ptr;
	typedef nnforge_shared_ptr<const layer> const_layer_smart_ptr;
	typedef std::vector<const_layer_smart_ptr> const_layer_list;
}
