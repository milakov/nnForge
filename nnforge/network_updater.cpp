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

#include "network_updater.h"

#include "neural_network_exception.h"
#include "rnd.h"
#include "nn_types.h"

#include <boost/format.hpp>

namespace nnforge
{
	const unsigned int network_updater::random_list_bits = 10;

	network_updater::network_updater(
		network_schema_smart_ptr schema,
		const_error_function_smart_ptr ef,
		const std::map<unsigned int, float>& layer_to_dropout_rate_map,
		const std::map<unsigned int, weight_vector_bound>& layer_to_weight_vector_bound_map,
		float weight_decay)
		: schema(schema)
		, ef(ef)
		, layer_to_dropout_rate_map(layer_to_dropout_rate_map)
		, random_uniform_list(1 << random_list_bits)
		, layer_to_weight_vector_bound_map(layer_to_weight_vector_bound_map)
		, weight_decay(weight_decay)
		, gen(rnd::get_random_generator())
	{
		const const_layer_list& layer_list = *schema;
		unsigned int layer_count = static_cast<unsigned int>(layer_list.size());

		std::vector<const_layer_smart_ptr>::const_iterator layer_it = schema->get_layers().begin();
		for(std::map<unsigned int, float>::const_iterator it = layer_to_dropout_rate_map.begin(); it != layer_to_dropout_rate_map.end(); ++it, ++layer_it)
		{
			if (it->first >= layer_count)
				throw neural_network_exception("Dropout is specified for the layer which doesn't exist in the schema");

			float dropout_rate = it->second;

			if ((dropout_rate < 0.0F) || (dropout_rate >= 1.0F))
				throw neural_network_exception((boost::format("Illegal dropout rate: %1%") % dropout_rate).str());

			if (dropout_rate > 0.0F)
			{
				dropout_layer_config mult = (schema->get_layers()[it->first])->get_dropout_layer_config(dropout_rate);
					layer_id_to_dropout_config_map.insert(std::make_pair(it->first, mult));
			}
		}

		for(std::map<unsigned int, weight_vector_bound>::iterator it = this->layer_to_weight_vector_bound_map.begin(); it != this->layer_to_weight_vector_bound_map.end(); ++it)
		{
			if (it->first >= layer_count)
				throw neural_network_exception("Weight bound is specified for the layer which doesn't exist in the schema");

			std::map<unsigned int, dropout_layer_config>::const_iterator it2 = layer_id_to_dropout_config_map.find(it->first);
			if (it2 != layer_id_to_dropout_config_map.end())
			{
				float val = 1.0F;
				for(std::map<unsigned int, float>::const_iterator it3 = it2->second.weight_part_to_dropout_direct_multiplier_map.begin(); it3 != it2->second.weight_part_to_dropout_direct_multiplier_map.end(); ++it3)
					val = std::min<float>(val, it3->second);
				it->second.max_l2_norm = it->second.max_l2_norm / val;
			}
		}
	}

	network_updater::~network_updater()
	{
	}

	void network_updater::set_input_configuration_specific(const layer_configuration_specific& input_configuration_specific)
	{
		if ((layer_config_list.size() > 0) && (layer_config_list[0] == input_configuration_specific))
			return;

		layer_config_list = schema->get_layer_configuration_specific_list(input_configuration_specific);

		update_flops();

		layer_config_list_modified();
	}

	std::vector<testing_result_smart_ptr> network_updater::update(
		supervised_data_reader& reader,
		const std::vector<network_data_smart_ptr>& learning_rate_vector_list,
		std::vector<network_data_smart_ptr>& data_list)
	{
		// Check data-schema consistency
		for(std::vector<network_data_smart_ptr>::iterator it = data_list.begin(); it != data_list.end(); it++)
			(*it)->check_network_data_consistency(*schema);
		for(std::vector<network_data_smart_ptr>::const_iterator it = learning_rate_vector_list.begin(); it != learning_rate_vector_list.end(); it++)
			(*it)->check_network_data_consistency(*schema);

		set_input_configuration_specific(reader.get_input_configuration());

		// Check schema-reader consistency
		layer_config_list[layer_config_list.size() - 1].check_equality(reader.get_output_configuration());

		nnforge_uniform_real_distribution<float> dist(0.0F, 1.0F);
		for(std::vector<float>::iterator it = random_uniform_list.begin(); it != random_uniform_list.end(); ++it)
			*it = dist(gen);

		for(std::vector<network_data_smart_ptr>::iterator it = data_list.begin(); it != data_list.end(); ++it)
			(*it)->apply_dropout_layer_config(layer_id_to_dropout_config_map, false);

		std::vector<testing_result_smart_ptr> res = actual_update(reader, learning_rate_vector_list, data_list);

		for(std::vector<network_data_smart_ptr>::iterator it = data_list.begin(); it != data_list.end(); ++it)
			(*it)->apply_dropout_layer_config(layer_id_to_dropout_config_map, true);

		return res;
	}

	void network_updater::update_flops()
	{
		flops = 0.0F;
		const const_layer_list& layer_list = *schema;
		bool non_empty_data_encountered = false;
		for(unsigned int i = 0; i < layer_list.size(); i++)
		{
			const_layer_smart_ptr layer = layer_list[i];
			const layer_configuration_specific& layer_conf = layer_config_list[i];

			flops += layer->get_forward_flops(layer_conf);

			if (non_empty_data_encountered)
				flops += layer->get_backward_flops(layer_conf);

			non_empty_data_encountered = non_empty_data_encountered || (!layer->is_empty_data());

			flops += layer->get_weights_update_flops(layer_conf);
		}
	}

	float network_updater::get_flops_for_single_entry() const
	{
		return flops;
	}
}
