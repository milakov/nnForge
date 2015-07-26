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

#include "layer_data_custom_list.h"

#include "neural_network_exception.h"

#include <boost/format.hpp>

namespace nnforge
{
	layer_data_custom_list::layer_data_custom_list()
	{
	}

	layer_data_custom_list::layer_data_custom_list(const std::vector<layer::const_ptr>& layer_list)
	{
		instance_name_to_data_custom_map.clear();
		for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
		{
			layer::const_ptr l = *it;
			layer_data_custom::ptr data_custom = l->create_layer_data_custom();
			if (!data_custom->empty())
				add(l->instance_name, data_custom);
		}
	}

	layer_data_custom_list::layer_data_custom_list(
		const std::vector<layer::const_ptr>& layer_list,
		const layer_data_custom_list& other)
	{
		for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
		{
			layer::const_ptr l = *it;
			if (!l->is_empty_data_custom())
			{
				layer_data_custom::ptr data_custom = other.find(l->instance_name);
				if (data_custom)
					add(l->instance_name, data_custom);
			}
		}
	}

	void layer_data_custom_list::check_consistency(const std::vector<layer::const_ptr>& layer_list) const
	{
		for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
		{
			layer::const_ptr l = *it;
			if (!l->is_empty_data_custom())
			{
				layer_data_custom::ptr data_custom = find(l->instance_name);
				if (!data_custom)
					throw neural_network_exception((boost::format("No custom data for layer with name %1%") % l->instance_name).str());
			
				l->check_layer_data_custom_consistency(*data_custom);
			}
		}
	}

	layer_data_custom::ptr layer_data_custom_list::find(const std::string& instance_name) const
	{
		std::map<std::string, layer_data_custom::ptr>::const_iterator it = instance_name_to_data_custom_map.find(instance_name);

		if (it != instance_name_to_data_custom_map.end())
			return it->second;
		else
			return layer_data_custom::ptr();
	}

	layer_data_custom::ptr layer_data_custom_list::get(const std::string& instance_name) const
	{
		layer_data_custom::ptr res = find(instance_name);

		if (!res)
			throw neural_network_exception((boost::format("No custom data found with layer name %1%") % instance_name).str());

		return res;
	}

	void layer_data_custom_list::write(std::ostream& binary_stream_to_write_to) const
	{
		unsigned int elem_count = static_cast<unsigned int>(instance_name_to_data_custom_map.size());
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&elem_count), sizeof(elem_count));
		for(std::map<std::string, layer_data_custom::ptr>::const_iterator it = instance_name_to_data_custom_map.begin(); it != instance_name_to_data_custom_map.end(); ++it)
		{
			unsigned int name_length = static_cast<unsigned int>(it->first.length());
			binary_stream_to_write_to.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
			binary_stream_to_write_to.write(it->first.data(), name_length);
			it->second->write(binary_stream_to_write_to);
		}
	}

	void layer_data_custom_list::read(std::istream& binary_stream_to_read_from)
	{
		instance_name_to_data_custom_map.clear();
		unsigned int elem_count;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&elem_count), sizeof(elem_count));
		for(unsigned int i = 0; i < elem_count; ++i)
		{
			unsigned int name_length;
			binary_stream_to_read_from.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
			std::string instance_name;
			instance_name.resize(name_length);
			binary_stream_to_read_from.read(&instance_name[0], name_length);
			layer_data_custom::ptr d(new layer_data_custom());
			d->read(binary_stream_to_read_from);
			add(instance_name, d);
		}
	}

	void layer_data_custom_list::add(
		const std::string& instance_name,
		layer_data_custom::ptr data_custom)
	{
		if (!instance_name_to_data_custom_map.insert(std::make_pair(instance_name, data_custom)).second)
			throw neural_network_exception((boost::format("Custom data with duplicate layer name %1%") % instance_name).str());
	}
}