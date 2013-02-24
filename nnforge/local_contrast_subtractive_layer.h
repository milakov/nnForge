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

#include "layer.h"

#include <vector>

// http://en.wikipedia.org/wiki/Discrete_Gaussian_kernel
namespace nnforge
{
	class local_contrast_subtractive_layer : public layer
	{
	public:
		local_contrast_subtractive_layer(
			const std::vector<unsigned int>& window_sizes,
			const std::vector<unsigned int>& feature_maps_affected,
			unsigned int feature_map_count);

		virtual layer_smart_ptr clone() const;

		virtual layer_configuration get_layer_configuration(const layer_configuration& input_configuration) const;

		virtual layer_configuration_specific get_layer_configuration_specific(const layer_configuration_specific& input_configuration_specific) const;

		virtual float get_forward_flops(const layer_configuration_specific& input_configuration_specific) const;

		virtual float get_backward_flops(const layer_configuration_specific& input_configuration_specific) const;

		virtual float get_backward_flops_2nd(const layer_configuration_specific& input_configuration_specific) const;

		virtual const boost::uuids::uuid& get_uuid() const;

		virtual void write(std::ostream& binary_stream_to_write_to) const;

		virtual void read(std::istream& binary_stream_to_read_from);

		static const boost::uuids::uuid layer_guid;

	public:
		std::vector<unsigned int> window_sizes;
		std::vector<unsigned int> feature_maps_affected;
		std::vector<unsigned int> feature_maps_unaffected;
		unsigned int feature_map_count;

		std::vector<std::vector<float> > window_weights_list;

	private:
		static const float c;

		// returns -1 in case the corresponding window size equals 1
		float get_std_dev(unsigned int dimension_id) const;

		float get_gaussian_value(
			int offset,
			unsigned int dimension_id) const;

		void setup_window_weights_list();
	};
}
