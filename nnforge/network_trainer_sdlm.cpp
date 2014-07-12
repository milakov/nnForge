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

#include "network_trainer_sdlm.h"

#include <boost/format.hpp>
#include <numeric>
#include <limits>
#include <fstream>

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
		, per_layer_mu(false)
	{
	}

	network_trainer_sdlm::~network_trainer_sdlm()
	{
	}

	void network_trainer_sdlm::train_step(
		supervised_data_reader& reader,
		training_task_state& task)
	{
		boost::chrono::steady_clock::time_point start = boost::chrono::high_resolution_clock::now();

		unsigned int hessian_entry_to_process_count = std::min<unsigned int>(std::max<unsigned int>(static_cast<unsigned int>(hessian_entry_to_process_ratio * reader.get_entry_count()), min_hessian_entry_to_process_count), reader.get_entry_count());

		network_data_smart_ptr learning_rate;

		network_data_smart_ptr hessian = hessian_calc->get_hessian(
			reader,
			task.data,
			hessian_entry_to_process_count);

		std::string comment = convert_hessian_to_training_vector(
			hessian,
			task.get_current_epoch());
		task.comments.push_back(comment);

		learning_rate = hessian;

		testing_result_smart_ptr train_result = updater->update(
			reader,
			learning_rate,
			task.data,
			batch_size,
			weight_decay,
			momentum);

		boost::chrono::duration<float> sec = (boost::chrono::high_resolution_clock::now() - start);

		float flops = updater->get_flops_for_single_entry();
		float flops_hessian = hessian_calc->get_flops_for_single_entry();

		train_result->time_to_complete_seconds = sec.count();
		train_result->flops = (static_cast<float>(train_result->get_entry_count()) * flops) + (static_cast<float>(hessian_entry_to_process_count) * flops_hessian);

		task.history.push_back(train_result);
	}

#ifdef NNFORGE_DEBUG_HESSIAN
	void network_trainer_sdlm::dump_lists(
		network_data_smart_ptr hessian,
		const char * filename_prefix) const
	{
		for(network_data::const_iterator it = hessian->begin(); it != hessian->end(); it++)
		{
			for(layer_data::const_iterator it2 = (*it)->begin(); it2 != (*it)->end(); it2++)
			{
				if (!it2->empty())
				{
					std::string filename = (boost::format("%1%_%|2$02d|_%|3$02d|.txt") % filename_prefix % (it - hessian->begin()) % (it2 - (*it)->begin())).str();
					std::ofstream out(filename.c_str());
					for(std::vector<float>::const_iterator it3 = it2->begin(); it3 != it2->end(); ++it3)
						out << *it3 << std::endl;
				}
			}
		}
	}
#endif

	std::string network_trainer_sdlm::convert_hessian_to_training_vector(
		network_data_smart_ptr hessian,
		unsigned int epoch_id) const
	{
#ifdef NNFORGE_DEBUG_HESSIAN
		dump_lists(hessian, "hessian");
#endif

		std::vector<std::vector<float> > average_hessian_list = get_average_hessian_list(hessian);
		std::string average_hessian_str;
		for(std::vector<std::vector<float> >::const_iterator it = average_hessian_list.begin(); it != average_hessian_list.end(); it++)
		{
			if (it != average_hessian_list.begin())
				average_hessian_str += ", ";

			for(std::vector<float>::const_iterator it2 = it->begin(); it2 != it->end(); it2++)
			{
				if (it2 != it->begin())
					average_hessian_str += " ";
				average_hessian_str += (boost::format("%|1$.1e|") % *it2).str();
			}
		}

		std::string convertion_str = per_layer_mu ?
			convert_hessian_to_training_vector_per_layer_mu(hessian, average_hessian_list, epoch_id) : convert_hessian_to_training_vector(hessian, average_hessian_list, epoch_id);

#ifdef NNFORGE_DEBUG_HESSIAN
		dump_lists(hessian, "learning_rate");
#endif

		return (boost::format("%|1$s|, Hessian (%|2$s|)") % convertion_str % average_hessian_str).str();
	}

	std::vector<std::vector<float> > network_trainer_sdlm::get_average_hessian_list(network_data_smart_ptr hessian) const
	{
		std::vector<std::vector<float> >res;

		float min_hessian = std::numeric_limits<float>::max();
		float max_hessian = std::numeric_limits<float>::min();

		for(network_data::iterator it = hessian->begin(); it != hessian->end(); it++)
		{
			if (!(*it)->empty())
			{
				std::vector<float> hs_list;
				for(layer_data::iterator it2 = (*it)->begin(); it2 != (*it)->end(); it2++)
				{
					float sum = std::accumulate(it2->begin(), it2->end(), 0.0F);
					float new_hessian_per_block = sum / it2->size();
					hs_list.push_back(new_hessian_per_block);
					min_hessian = std::min<float>(min_hessian, new_hessian_per_block);
					max_hessian = std::max<float>(max_hessian, new_hessian_per_block);
				}
				res.push_back(hs_list);
			}
		}

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



	std::string network_trainer_sdlm::convert_hessian_to_training_vector_per_layer_mu(
		network_data_smart_ptr hessian,
		const std::vector<std::vector<float> >& average_hessian_list,
		unsigned int epoch_id) const
	{
		std::vector<std::vector<float> >::const_iterator ah_it = average_hessian_list.begin();
		std::vector<std::vector<float> > avg_lr_lists;
		for(network_data::iterator it = hessian->begin(); it != hessian->end(); it++)
		{
			if ((*it)->size() > 0)
			{
				std::vector<float>::const_iterator ah_it2 = ah_it->begin();
				std::vector<float> avg_lr_list;
				for(layer_data::iterator it2 = (*it)->begin(); it2 != (*it)->end(); it2++, ah_it2++)
				{
					float mu = *ah_it2;
					float eta = mu * get_global_learning_rate(static_cast<unsigned int>(epoch_id));
					hessian_transform ht(mu, eta);
					std::transform(it2->begin(), it2->end(), it2->begin(), ht);

					float sum = std::accumulate(it2->begin(), it2->end(), 0.0F);
					float new_vg_lr = sum / it2->size();
					avg_lr_list.push_back(new_vg_lr);
				}
				avg_lr_lists.push_back(avg_lr_list);
				++ah_it;
			}
		}

		std::string average_lr_str;
		for(std::vector<std::vector<float> >::const_iterator it = avg_lr_lists.begin(); it != avg_lr_lists.end(); it++)
		{
			if (it != avg_lr_lists.begin())
				average_lr_str += ", ";

			for(std::vector<float>::const_iterator it2 = it->begin(); it2 != it->end(); it2++)
			{
				if (it2 != it->begin())
					average_lr_str += " ";
				average_lr_str += (boost::format("%|1$.1e|") % *it2).str();
			}
		}

		return (boost::format("LR (%|1$s|)") % average_lr_str).str();
	}

	std::string network_trainer_sdlm::convert_hessian_to_training_vector(
		network_data_smart_ptr hessian,
		const std::vector<std::vector<float> >& average_hessian_list,
		unsigned int epoch_id) const
	{
		float min_hessian = std::numeric_limits<float>::max();
		float max_hessian = std::numeric_limits<float>::min();
		for(std::vector<std::vector<float> >::const_iterator it = average_hessian_list.begin(); it != average_hessian_list.end(); ++it)
		{
			const std::vector<float>& avl = *it;
			std::vector<float>::const_iterator it_max = std::max_element(avl.begin(), avl.end());
			if (it_max != avl.end())
				max_hessian = std::max(max_hessian, *it_max);
			std::vector<float>::const_iterator it_min = std::min_element(avl.begin(), avl.end());
			if (it_min != avl.end())
				min_hessian = std::min(min_hessian, *it_min);
		}
		float max_mu_current = std::min(max_mu, max_hessian * 0.5F);
		float mu = min_hessian * 0.5F * powf(mu_increase_factor, static_cast<float>(epoch_id));
		mu = std::min(mu, max_mu_current);

		float eta = mu * get_global_learning_rate(static_cast<unsigned int>(epoch_id));

		std::vector<std::vector<float> > avg_lr_lists;
		for(network_data::iterator it = hessian->begin(); it != hessian->end(); it++)
		{
			if ((*it)->size() > 0)
			{
				std::vector<float> avg_lr_list;
				for(layer_data::iterator it2 = (*it)->begin(); it2 != (*it)->end(); it2++)
				{
					hessian_transform ht(mu, eta);
					std::transform(it2->begin(), it2->end(), it2->begin(), ht);

					float sum = std::accumulate(it2->begin(), it2->end(), 0.0F);
					float new_vg_lr = sum / it2->size();
					avg_lr_list.push_back(new_vg_lr);
				}
				avg_lr_lists.push_back(avg_lr_list);
			}
		}

		std::string average_lr_str;
		for(std::vector<std::vector<float> >::const_iterator it = avg_lr_lists.begin(); it != avg_lr_lists.end(); it++)
		{
			if (it != avg_lr_lists.begin())
				average_lr_str += ", ";

			for(std::vector<float>::const_iterator it2 = it->begin(); it2 != it->end(); it2++)
			{
				if (it2 != it->begin())
					average_lr_str += " ";
				average_lr_str += (boost::format("%|1$.1e|") % *it2).str();
			}
		}

		return (boost::format("Eta = %|1$.2e|, Mu = %|2$.2e|, LR (%|3$s|)") % eta % mu % average_lr_str).str();
	}

	void network_trainer_sdlm::initialize_train(supervised_data_reader& reader)
	{
		updater->set_input_configuration_specific(reader.get_input_configuration());
		hessian_calc->set_input_configuration_specific(reader.get_input_configuration());
	}
}
