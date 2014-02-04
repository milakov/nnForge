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

#include "network_trainer_sdlm.h"

#include <boost/format.hpp>
#include <numeric> 

#include "neural_network_exception.h"

namespace nnforge
{
	const unsigned int network_trainer_sdlm::min_hessian_entry_to_process_count = 10;

	network_trainer_sdlm::network_trainer_sdlm(
		network_schema_smart_ptr schema,
		hessian_calculator_smart_ptr hessian_calc,
		network_updater_smart_ptr updater)
		: network_trainer(schema)
		, hessian_calc(hessian_calc)
		, updater(updater)
		, hessian_entry_to_process_ratio(0.05F)
		, max_mu(1.0F)
		, mu_increase_factor(1.0F)
		, speed(0.02F)
	{
	}

	network_trainer_sdlm::~network_trainer_sdlm()
	{
	}

	void network_trainer_sdlm::train_step(
		supervised_data_reader& reader,
		std::vector<training_task_state>& task_list)
	{
		boost::chrono::steady_clock::time_point start = boost::chrono::high_resolution_clock::now();

		unsigned int hessian_entry_to_process_count = std::min<unsigned int>(std::max<unsigned int>(static_cast<unsigned int>(hessian_entry_to_process_ratio * reader.get_entry_count()), min_hessian_entry_to_process_count), reader.get_entry_count());

		std::vector<network_data_smart_ptr> learning_rate_vector_list;
		for(unsigned int i = 0; i < task_list.size(); ++i)
		{
			network_data_smart_ptr hessian = hessian_calc->get_hessian(
				reader,
				task_list[i].data,
				hessian_entry_to_process_count);

			std::string comment = convert_hessian_to_training_vector(
				hessian,
				task_list[i].history);

			learning_rate_vector_list.push_back(hessian);

			task_list[i].comments.push_back(comment);
		}

		std::vector<network_data_smart_ptr> data_list;
		for(std::vector<training_task_state>::iterator it = task_list.begin(); it != task_list.end(); ++it)
			data_list.push_back(it->data);

		std::vector<testing_result_smart_ptr> train_result = updater->update(
			reader,
			learning_rate_vector_list,
			data_list);

		boost::chrono::duration<float> sec = (boost::chrono::high_resolution_clock::now() - start) / task_list.size();

		float flops = updater->get_flops_for_single_entry();
		float flops_hessian = hessian_calc->get_flops_for_single_entry();

		for(unsigned int i = 0; i < task_list.size(); ++i)
		{
			testing_result_smart_ptr res = train_result[i];
			res->time_to_complete_seconds = sec.count();
			res->flops = (static_cast<float>(res->entry_count) * flops) + (static_cast<float>(hessian_entry_to_process_count) * flops_hessian);

			task_list[i].history.push_back(res);
		}
	}

	std::string network_trainer_sdlm::convert_hessian_to_training_vector(
		network_data_smart_ptr hessian,
		const std::vector<testing_result_smart_ptr>& history) const
	{
		std::vector<std::vector<float> > original_mu_list;

		float mu = get_mu(
			hessian,
			history,
			original_mu_list);

		float eta = get_eta(
			mu,
			history);

		convert_hessian_to_training_vector(hessian, mu, eta);

		std::string original_mu_str;
		for(std::vector<std::vector<float> >::const_iterator it = original_mu_list.begin(); it != original_mu_list.end(); it++)
		{
			if (it != original_mu_list.begin())
				original_mu_str += ", ";

			for(std::vector<float>::const_iterator it2 = it->begin(); it2 != it->end(); it2++)
			{
				if (it2 != it->begin())
					original_mu_str += " ";
				original_mu_str += (boost::format("%|1$.1e|") % *it2).str();
			}
		}

		return (boost::format("Eta = %|1$.2e|, Mu = %|2$.2e| (%|3$s|)") % eta % mu % original_mu_str).str();
	}

	float network_trainer_sdlm::get_mu(
		network_data_smart_ptr hessian,
		const std::vector<testing_result_smart_ptr>& history,
		std::vector<std::vector<float> >& original_mu_list) const
	{
		original_mu_list.clear();

		float min_hessian = 1.0e38F;
		float max_hessian = 1.0e-37F;

		for(network_data::iterator it = hessian->begin(); it != hessian->end(); it++)
		{
			if ((*it)->size() > 0)
			{
				std::vector<float> mu_list;
				for(layer_data::iterator it2 = (*it)->begin(); it2 != (*it)->end(); it2++)
				{
					float sum = std::accumulate(it2->begin(), it2->end(), 0.0F);
					float new_hessian_per_block = sum / it2->size();
					mu_list.push_back(new_hessian_per_block);
					min_hessian = std::min<float>(min_hessian, new_hessian_per_block);
					max_hessian = std::max<float>(max_hessian, new_hessian_per_block);
				}
				original_mu_list.push_back(mu_list);
			}
		}

		float max_mu_current = std::min<float>(max_mu, max_hessian * 0.5F);
		float current_mu = min_hessian * 0.5F * powf(mu_increase_factor, static_cast<float>(history.size()));
		current_mu = std::min<float>(current_mu, max_mu_current);
		return current_mu;
	}

	float network_trainer_sdlm::get_eta(
		float mu,
		const std::vector<testing_result_smart_ptr>& history) const
	{
		float res = mu * speed;

		res *= get_tail_decay_factor(static_cast<unsigned int>(history.size()));

		return res;
	}

	network_trainer_sdlm::hessian_transform::hessian_transform(float mu, float eta)
		: mu(mu)
		, eta(eta)
	{
	}
			
	float network_trainer_sdlm::hessian_transform::operator() (float in)
	{
		return eta / (in + mu);
	}

	void network_trainer_sdlm::convert_hessian_to_training_vector(
		network_data_smart_ptr hessian,
		float mu,
		float eta) const
	{
		hessian_transform ht(mu, eta);

		for(network_data::iterator it = hessian->begin(); it != hessian->end(); it++)
			for(layer_data::iterator it2 = (*it)->begin(); it2 != (*it)->end(); it2++)
				std::transform(it2->begin(), it2->end(), it2->begin(), ht);
	}

	unsigned int network_trainer_sdlm::get_max_batch_size() const
	{
		return updater->get_max_batch_size();
	}

	void network_trainer_sdlm::initialize_train(supervised_data_reader& reader)
	{
		updater->set_input_configuration_specific(reader.get_input_configuration());
	}
}
