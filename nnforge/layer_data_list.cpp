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

#include "layer_data_list.h"

#include "neural_network_exception.h"

#include <numeric>
#include <boost/format.hpp>
#include <cmath>

namespace nnforge
{
	layer_data_list::layer_data_list()
	{
	}

	layer_data_list::layer_data_list(
		const std::vector<layer::const_ptr>& layer_list,
		float val)
	{
		for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
		{
			layer::const_ptr l = *it;
			layer_data::ptr data = l->create_layer_data();
			if (!data->empty())
			{
				add(l->instance_name, data);
				data->fill(val);
			}
		}
	}

	layer_data_list::layer_data_list(
		const std::vector<layer::const_ptr>& layer_list,
		const layer_data_list& other)
	{
		for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
		{
			layer::const_ptr l = *it;
			if (!l->is_empty_data())
			{
				layer_data::ptr data = other.find(l->instance_name);
				if (data)
					add(l->instance_name, data);
			}
		}
	}

	void layer_data_list::check_consistency(const std::vector<layer::const_ptr>& layer_list) const
	{
		for(std::vector<layer::const_ptr>::const_iterator it = layer_list.begin(); it != layer_list.end(); ++it)
		{
			layer::const_ptr l = *it;
			if (!l->is_empty_data())
			{
				layer_data::ptr data = find(l->instance_name);
				if (!data)
					throw neural_network_exception((boost::format("No data for layer with name %1%") % l->instance_name).str());
			
				l->check_layer_data_consistency(*data);
			}
		}
	}

	std::string layer_data_list::get_stat() const
	{
		std::string stat = "Avg Abs weights";

		for(std::map<std::string, layer_data::ptr>::const_iterator it = instance_name_to_data_map.begin(); it != instance_name_to_data_map.end(); ++it)
		{
			std::string layer_stat;

			for(layer_data::const_iterator it2 = it->second->begin(); it2 != it->second->end(); it2++)
			{
				const std::vector<float>& data = *it2;

				double sum = 0.0;
				for(std::vector<float>::const_iterator it3 = data.begin(); it3 != data.end(); ++it3)
					sum += static_cast<float>(fabsf(*it3));
				float avg = static_cast<float>(sum) / static_cast<float>(data.size());

				if (!layer_stat.empty())
					layer_stat += ", ";
				layer_stat += (boost::format("%|1$.5e|") % avg).str();
			}

			if (!layer_stat.empty())
			{
				stat += (boost::format(" %1% (%2%)") % it->first % layer_stat).str();
			}
		}

		return stat;
	}

	void layer_data_list::fill(float val)
	{
		for(std::map<std::string, layer_data::ptr>::iterator it = instance_name_to_data_map.begin(); it != instance_name_to_data_map.end(); ++it)
			it->second->fill(val);
	}

	void layer_data_list::random_fill(
		float min,
		float max,
		random_generator& gen)
	{
		for(std::map<std::string, layer_data::ptr>::iterator it = instance_name_to_data_map.begin(); it != instance_name_to_data_map.end(); ++it)
			it->second->random_fill(min, max, gen);
	}

	layer_data::ptr layer_data_list::find(const std::string& instance_name) const
	{
		std::map<std::string, layer_data::ptr>::const_iterator it = instance_name_to_data_map.find(instance_name);

		if (it != instance_name_to_data_map.end())
			return it->second;
		else
			return layer_data::ptr();
	}

	layer_data::ptr layer_data_list::get(const std::string& instance_name) const
	{
		layer_data::ptr res = find(instance_name);

		if (!res)
			throw neural_network_exception((boost::format("No data found with layer name %1%") % instance_name).str());

		return res;
	}

	void layer_data_list::write(std::ostream& binary_stream_to_write_to) const
	{
		unsigned int elem_count = static_cast<unsigned int>(instance_name_to_data_map.size());
		binary_stream_to_write_to.write(reinterpret_cast<const char*>(&elem_count), sizeof(elem_count));
		for(std::map<std::string, layer_data::ptr>::const_iterator it = instance_name_to_data_map.begin(); it != instance_name_to_data_map.end(); ++it)
		{
			unsigned int name_length = static_cast<unsigned int>(it->first.length());
			binary_stream_to_write_to.write(reinterpret_cast<const char*>(&name_length), sizeof(name_length));
			binary_stream_to_write_to.write(it->first.data(), name_length);
			it->second->write(binary_stream_to_write_to);
		}
	}

	void layer_data_list::read(std::istream& binary_stream_to_read_from)
	{
		instance_name_to_data_map.clear();
		unsigned int elem_count;
		binary_stream_to_read_from.read(reinterpret_cast<char*>(&elem_count), sizeof(elem_count));
		for(unsigned int i = 0; i < elem_count; ++i)
		{
			unsigned int name_length;
			binary_stream_to_read_from.read(reinterpret_cast<char*>(&name_length), sizeof(name_length));
			std::string instance_name;
			instance_name.resize(name_length);
			binary_stream_to_read_from.read(&instance_name[0], name_length);
			layer_data::ptr d(new layer_data());
			d->read(binary_stream_to_read_from);
			add(instance_name, d);
		}
	}

	void layer_data_list::add(
		const std::string& instance_name,
		layer_data::ptr data)
	{
		if (!instance_name_to_data_map.insert(std::make_pair(instance_name, data)).second)
			throw neural_network_exception((boost::format("Data with duplicate layer name %1%") % instance_name).str());
	}
}
