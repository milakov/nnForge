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

#include "untile_layer.h"

#include "layer_factory.h"
#include "neural_network_exception.h"
#include "proto/nnforge.pb.h"

#include <algorithm>
#include <boost/lambda/lambda.hpp>
#include <boost/format.hpp>

namespace nnforge
{
	// {385C07DA-B164-4724-A646-27763E71792F}
	const boost::uuids::uuid untile_layer::layer_guid =
	{ 0x38, 0x5c, 0x07, 0xda
		, 0xb1, 0x64
		, 0x47, 0x24
		, 0xa6, 0x46
		, 0x27, 0x76, 0x3e, 0x71, 0x79, 0x2f };

	const std::string untile_layer::layer_type_name = "UnTile";

	untile_layer::untile_layer(const std::vector<std::vector<unsigned int> >& upsampling_sizes_list)
		: upsampling_sizes_list(upsampling_sizes_list)
	{
		check();
	}

	void untile_layer::check()
	{
		if (upsampling_sizes_list.size() == 0)
			throw neural_network_exception("level list for untile layer may not be empty");

		unsigned int dimension_count = static_cast<unsigned int>(upsampling_sizes_list.front().size());
		if (dimension_count == 0)
			throw neural_network_exception("upsampling size for untile layer may not be zero");
		for(unsigned int i = 1; i < static_cast<unsigned int>(upsampling_sizes_list.size()); i++)
		{
			unsigned int new_dimension_count = static_cast<unsigned int>(upsampling_sizes_list[i].size());
			if (dimension_count != new_dimension_count)
				throw neural_network_exception("window dimensions for max subsampling layer should be equal for all levels");
		}
	}

	const boost::uuids::uuid& untile_layer::get_uuid() const
	{
		return layer_guid;
	}

	const std::string& untile_layer::get_type_name() const
	{
		return layer_type_name;
	}

	layer_smart_ptr untile_layer::clone() const
	{
		return layer_smart_ptr(new untile_layer(*this));
	}

	layer_configuration untile_layer::get_layer_configuration(const layer_configuration& input_configuration) const
	{
		if ((input_configuration.dimension_count >= 0) && (input_configuration.dimension_count != static_cast<int>(upsampling_sizes_list.front().size())))
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % upsampling_sizes_list.front().size() % input_configuration.dimension_count).str());

		return layer_configuration(input_configuration.feature_map_count, static_cast<int>(upsampling_sizes_list.front().size()));
	}

	layer_configuration_specific untile_layer::get_output_layer_configuration_specific(const layer_configuration_specific& input_configuration_specific) const
	{
		if (input_configuration_specific.get_dimension_count() != upsampling_sizes_list.front().size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and input configuration (%2%) don't match") % upsampling_sizes_list.front().size() % input_configuration_specific.get_dimension_count()).str());

		layer_configuration_specific res(input_configuration_specific);

		for(unsigned int i = 0; i < upsampling_sizes_list.size(); ++i)
		{
			const std::vector<unsigned int>& upsampling_sizes = upsampling_sizes_list[i];
			for(unsigned int j = 0; j < upsampling_sizes.size(); ++j)
				res.dimension_sizes[j] *= upsampling_sizes[j];
		}

		return res;
	}

	layer_configuration_specific untile_layer::get_input_layer_configuration_specific(const layer_configuration_specific& output_configuration_specific) const
	{
		if (output_configuration_specific.get_dimension_count() != upsampling_sizes_list.front().size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and output configuration (%2%) don't match") % upsampling_sizes_list.front().size() % output_configuration_specific.get_dimension_count()).str());

		layer_configuration_specific res(output_configuration_specific);

		for(unsigned int i = 0; i < upsampling_sizes_list.size(); ++i)
		{
			const std::vector<unsigned int>& upsampling_sizes = upsampling_sizes_list[i];
			for(unsigned int j = 0; j < upsampling_sizes.size(); ++j)
			{
				if (res.dimension_sizes[j] % upsampling_sizes[j] == 0)
					res.dimension_sizes[j] /= upsampling_sizes[j];
				else
					throw neural_network_exception("upsampling sizes of untile layer cannot evenly divide output sizes");
			}
		}

		return res;
	}

	std::vector<std::pair<unsigned int, unsigned int> > untile_layer::get_input_rectangle_borders(const std::vector<std::pair<unsigned int, unsigned int> >& output_rectangle_borders) const
	{
		if (output_rectangle_borders.size() != upsampling_sizes_list.front().size())
			throw neural_network_exception((boost::format("Dimension count in layer (%1%) and output borders (%2%) don't match") % upsampling_sizes_list.front().size() % output_rectangle_borders.size()).str());

		std::vector<std::pair<unsigned int, unsigned int> > res = output_rectangle_borders;

		for(unsigned int i = 0; i < upsampling_sizes_list.size(); ++i)
		{
			const std::vector<unsigned int>& upsampling_sizes = upsampling_sizes_list[i];
			for(unsigned int j = 0; j < upsampling_sizes.size(); ++j)
			{
				res[j].first /= upsampling_sizes[j];
				res[j].second /= upsampling_sizes[j];
			}
		}

		return res;
	}

	void untile_layer::write(std::ostream& binary_stream_to_write_to) const
	{
		throw neural_network_exception("write() is not available for untile_layer");
	}

	void untile_layer::write_proto(void * layer_proto) const
	{
		protobuf::Layer * layer_proto_typed = reinterpret_cast<protobuf::Layer *>(layer_proto);
		protobuf::UnTileParam * param = layer_proto_typed->mutable_untile_param();

		for(int i = 0; i < upsampling_sizes_list.size(); ++i)
		{
			const std::vector<unsigned int>& upsampling_sizes = upsampling_sizes_list[i];
			protobuf::UnTileParam_UnTileLevelParam * level_param = param->add_level_param();
			for(int j = 0; j < upsampling_sizes.size(); ++j)
			{
				protobuf::UnTileParam_UnTileUpsamplingDimensionParam * dim_param = level_param->add_dimension_param();
				dim_param->set_upsampling_size(upsampling_sizes[j]);
			}
		}
	}

	void untile_layer::read(
		std::istream& binary_stream_to_read_from,
		const boost::uuids::uuid& layer_read_guid)
	{
		throw neural_network_exception("read() is not available for untile_layer");
	}

	void untile_layer::read_proto(const void * layer_proto)
	{
		const protobuf::Layer * layer_proto_typed = reinterpret_cast<const protobuf::Layer *>(layer_proto);
		if (!layer_proto_typed->has_untile_param())
			throw neural_network_exception((boost::format("No has_untile_param specified for layer %1% of type %2%") % instance_name % layer_proto_typed->type()).str());

		upsampling_sizes_list.resize(layer_proto_typed->untile_param().level_param_size());
		for(int i = 0; i < layer_proto_typed->untile_param().level_param_size(); ++i)
		{
			const protobuf::UnTileParam_UnTileLevelParam& level_param = layer_proto_typed->untile_param().level_param(i);
			std::vector<unsigned int>& upsampling_sizes = upsampling_sizes_list[i];
			upsampling_sizes.resize(level_param.dimension_param_size());
			for(int j = 0; j < level_param.dimension_param_size(); ++j)
				upsampling_sizes[j] = level_param.dimension_param(j).upsampling_size();
		}

		check();
	}

	float untile_layer::get_forward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(0);
	}

	float untile_layer::get_backward_flops(const layer_configuration_specific& input_configuration_specific) const
	{
		return static_cast<float>(0);
	}

	std::vector<tiling_factor> untile_layer::get_tiling_factor_list() const
	{
		std::vector<tiling_factor> res(upsampling_sizes_list.front().size(), 1);

		for(int i = 0; i < upsampling_sizes_list.size(); ++i)
		{
			const std::vector<unsigned int>& upsampling_sizes = upsampling_sizes_list[i];
			for(int j = 0; j < upsampling_sizes.size(); ++j)
				res[j] = res[j] * tiling_factor(upsampling_sizes[j], false);
		}

		return res;
	}
}
