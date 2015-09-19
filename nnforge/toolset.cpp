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

#include "toolset.h"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/format.hpp>
#include <iostream>

#include "neural_network_exception.h"
#include "neuron_value_set_data_bunch_writer.h"
#include "network_trainer_sgd.h"
#include "network_data_peeker_random.h"
#include "complex_network_data_pusher.h"
#include "save_resume_network_data_pusher.h"
#include "report_progress_network_data_pusher.h"
#include "summarize_network_data_pusher.h"
#include "validate_progress_network_data_pusher.h"

namespace nnforge
{
	const char * toolset::logfile_name = "log.txt";
	const char * toolset::ann_subfolder_name = "batch";
	const char * toolset::debug_subfolder_name = "debug";
	const char * toolset::trained_ann_index_extractor_pattern = "^ann_trained_(\\d+)\\.data$";
	const char * toolset::resume_ann_index_extractor_pattern = "^ann_trained_(\\d+)_epoch_(\\d+)\\.data$";
	const char * toolset::ann_resume_subfolder_name = "resume";

	toolset::toolset(factory_generator::ptr master_factory)
		: master_factory(master_factory)
	{
	}

	toolset::~toolset()
	{
	}

	void toolset::do_action()
	{
		if (!action.compare("info"))
		{
			master_factory->info();
		}
		else if (!action.compare("inference"))
		{
			run_inference();
		}
		else if (!action.compare("dump_schema_dot"))
		{
			dump_schema_dot();
		}
		else if (!action.compare("train"))
		{
			train();
		}
		else
		{
			do_custom_action();
		}
	}

	bool toolset::parse(int argc, char* argv[])
	{
		default_config_path = argv[0];
		default_config_path += ".cfg";

		// Declare a group of options that will be 
		// allowed only on command line
		boost::program_options::options_description gener("Generic options");
		gener.add_options()
			("help", "produce help message")
			;

		// Declare a group of options that will be 
		// allowed both on command line and in
		// config file
		boost::program_options::options_description config("Configuration");

		{
			std::vector<string_option> additional_string_options = get_string_options();
			for(std::vector<string_option>::iterator it = additional_string_options.begin(); it != additional_string_options.end(); it++)
			{
				string_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<std::string>(opt.var)->default_value(opt.default_value.c_str()), opt.description.c_str());
			}
			std::vector<multi_string_option> additional_multi_string_options = get_multi_string_options();
			for(std::vector<multi_string_option>::iterator it = additional_multi_string_options.begin(); it != additional_multi_string_options.end(); it++)
			{
				multi_string_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<std::vector<std::string> >(opt.var), opt.description.c_str());
			}
			std::vector<path_option> additional_path_options = get_path_options();
			for(std::vector<path_option>::iterator it = additional_path_options.begin(); it != additional_path_options.end(); it++)
			{
				path_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<boost::filesystem::path>(opt.var)->default_value(opt.default_value.c_str()), opt.description.c_str());
			}
			std::vector<bool_option> additional_bool_options = get_bool_options();
			for(std::vector<bool_option>::iterator it = additional_bool_options.begin(); it != additional_bool_options.end(); it++)
			{
				bool_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<bool>(opt.var)->default_value(opt.default_value), opt.description.c_str());
			}
			std::vector<float_option> additional_float_options = get_float_options();
			for(std::vector<float_option>::iterator it = additional_float_options.begin(); it != additional_float_options.end(); it++)
			{
				float_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<float>(opt.var)->default_value(opt.default_value), opt.description.c_str());
			}
			std::vector<int_option> additional_int_options = get_int_options();
			for(std::vector<int_option>::iterator it = additional_int_options.begin(); it != additional_int_options.end(); it++)
			{
				int_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<int>(opt.var)->default_value(opt.default_value), opt.description.c_str());
			}
		}

		{
			std::vector<string_option> additional_string_options = master_factory->get_string_options();
			for(std::vector<string_option>::iterator it = additional_string_options.begin(); it != additional_string_options.end(); it++)
			{
				string_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<std::string>(opt.var)->default_value(opt.default_value.c_str()), opt.description.c_str());
			}
			std::vector<multi_string_option> additional_multi_string_options = master_factory->get_multi_string_options();
			for(std::vector<multi_string_option>::iterator it = additional_multi_string_options.begin(); it != additional_multi_string_options.end(); it++)
			{
				multi_string_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<std::vector<std::string> >(opt.var), opt.description.c_str());
			}
			std::vector<path_option> additional_path_options = master_factory->get_path_options();
			for(std::vector<path_option>::iterator it = additional_path_options.begin(); it != additional_path_options.end(); it++)
			{
				path_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<boost::filesystem::path>(opt.var)->default_value(opt.default_value.c_str()), opt.description.c_str());
			}
			std::vector<bool_option> additional_bool_options = master_factory->get_bool_options();
			for(std::vector<bool_option>::iterator it = additional_bool_options.begin(); it != additional_bool_options.end(); it++)
			{
				bool_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<bool>(opt.var)->default_value(opt.default_value), opt.description.c_str());
			}
			std::vector<float_option> additional_float_options = master_factory->get_float_options();
			for(std::vector<float_option>::iterator it = additional_float_options.begin(); it != additional_float_options.end(); it++)
			{
				float_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<float>(opt.var)->default_value(opt.default_value), opt.description.c_str());
			}
			std::vector<int_option> additional_int_options = master_factory->get_int_options();
			for(std::vector<int_option>::iterator it = additional_int_options.begin(); it != additional_int_options.end(); it++)
			{
				int_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<int>(opt.var)->default_value(opt.default_value), opt.description.c_str());
			}
		}

		// Hidden options, will be allowed both on command line and
		// in config file, but will not be shown to the user.
		boost::program_options::options_description hidden("Hidden options");
		hidden.add_options()
			;

		boost::program_options::options_description cmdline_options;
		cmdline_options.add(gener).add(config).add(hidden);

		boost::program_options::options_description config_file_options;
		config_file_options.add(config).add(hidden);

		boost::program_options::options_description visible("Allowed options");
		visible.add(gener).add(config);

		boost::program_options::positional_options_description p;
		p.add("action", -1);

		boost::program_options::variables_map vm;
		boost::program_options::store(boost::program_options::command_line_parser(argc, argv).
				options(cmdline_options).positional(p).run(), vm);
		boost::program_options::notify(vm);

		boost::filesystem::ifstream ifs(config_file_path);
		if (!ifs)
			throw std::runtime_error((boost::format("Can not open config file %1%") % config_file_path).str());

		boost::program_options::store(parse_config_file(ifs, config_file_options, true), vm);
		boost::program_options::notify(vm);

		if (vm.count("help"))
		{
			std::cout << visible << "\n";
			return false;
		}

		boost::filesystem::path logfile_path = get_working_data_folder() / logfile_name;
		std::cout << "Forking output log to " << logfile_path.string() << "..." << std::endl;
		out_to_log_duplicator = nnforge_shared_ptr<stream_duplicator>(new stream_duplicator(logfile_path));

		{
			time_t rawtime;
			struct tm * timeinfo;
			char buffer[80];
			time(&rawtime);
			timeinfo = localtime(&rawtime);
			strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", timeinfo);
			std::cout << buffer << std::endl;
		}

		dump_settings();
		std::cout << "----------------------------------------" << std::endl;

		debug = debug_state::ptr(new debug_state(debug_mode, get_working_data_folder() / debug_subfolder_name));

		master_factory->initialize();

		forward_propagation_factory = master_factory->create_forward_propagation_factory();
		backward_propagation_factory = master_factory->create_backward_propagation_factory();

		return (action.size() > 0);
	}

	std::string toolset::get_default_action() const
	{
		return std::string();
	}

	std::string toolset::get_action() const
	{
		return action;
	}

	std::vector<string_option> toolset::get_string_options()
	{
		std::vector<string_option> res;

		res.push_back(string_option("action", &action, get_default_action().c_str(), "run action (info, run_inference, dump_schema_dot, train)"));
		res.push_back(string_option("schema", &schema_filename, "schema.txt", "Name of the file with schema of the network, in protobuf format"));
		res.push_back(string_option("inference_dataset_name", &inference_dataset_name, "validating", "Name of the dataset to be used for inference"));
		res.push_back(string_option("training_dataset_name", &training_dataset_name, "training_randomized", "Name of the dataset to be used for training"));
		res.push_back(string_option("training_algo", &training_algo, "", "Training algorithm (sgd)"));
		res.push_back(string_option("momentum_type", &momentum_type_str, "vanilla", "Type of the momentum to use (none, vanilla, nesterov)"));

		return res;
	}

	std::vector<multi_string_option> toolset::get_multi_string_options()
	{
		std::vector<multi_string_option> res;

		res.push_back(multi_string_option("inference_output_layer_name", &inference_output_layer_names, "Names of the output layers when doing inference"));
		res.push_back(multi_string_option("training_output_layer_name", &training_output_layer_names, "Names of the output layers when doing training"));
		res.push_back(multi_string_option("training_error_source_layer_name", &training_error_source_layer_names, "Names of the error sources for training"));
		res.push_back(multi_string_option("training_exclude_data_update_layer_name", &training_exclude_data_update_layer_names, "Names of layers which shouldn't be trained"));

		return res;
	}

	std::vector<path_option> toolset::get_path_options()
	{
		std::vector<path_option> res;

		res.push_back(path_option("config", &config_file_path, default_config_path.c_str(), "Path to the configuration file"));
		res.push_back(path_option("working_data_folder", &working_data_folder, "", "Path to the folder where data are processed"));

		return res;
	}

	std::vector<bool_option> toolset::get_bool_options()
	{
		std::vector<bool_option> res;

		res.push_back(bool_option("debug_mode", &debug_mode, false, "Debug mode"));
		res.push_back(bool_option("load_resume,R", &load_resume, false, "Resume neural network training strating from saved state"));
		res.push_back(bool_option("dump_resume,R", &dump_resume, true, "Dump neural network data after each epoch"));

		return res;
	}

	std::vector<float_option> toolset::get_float_options()
	{
		std::vector<float_option> res;

		res.push_back(float_option("learning_rate,L", &learning_rate, 0.01F, "Global learning rate"));
		res.push_back(float_option("learning_rate_decay_rate", &learning_rate_decay_rate, 0.5F, "Degradation of learning rate at each tail epoch"));
		res.push_back(float_option("learning_rate_rise_rate", &learning_rate_rise_rate, 0.1F, "Increase factor of learning rate at each head epoch (<1.0)"));
		res.push_back(float_option("weight_decay", &weight_decay, 0.0F, "Weight decay"));
		res.push_back(float_option("momentum,M", &momentum_val, 0.0F, "Momentum value"));

		return res;
	}

	std::vector<int_option> toolset::get_int_options()
	{
		std::vector<int_option> res;

		res.push_back(int_option("training_epoch_count,E", &training_epoch_count, 50, "Epochs to train"));
		res.push_back(int_option("learning_rate_decay_tail", &learning_rate_decay_tail_epoch_count, 0, "Number of tail iterations with gradually lowering learning rates"));
		res.push_back(int_option("learning_rate_rise_head", &learning_rate_rise_head_epoch_count, 0, "Number of head iterations with gradually increasing learning rates"));
		res.push_back(int_option("batch_size,B", &batch_size, 1, "Training mini-batch size"));
		res.push_back(int_option("ann_count,N", &ann_count, 1, "Amount of networks to train"));
		res.push_back(int_option("inference_ann_data_index", &inference_ann_data_index, -1, "Index of the dataset to be used for inference"));
		res.push_back(int_option("batch_offset", &batch_offset, 0, "Shift initial ANN index when batch training"));

		return res;
	}

	boost::filesystem::path toolset::get_working_data_folder() const
	{
		return working_data_folder;
	}

	void toolset::dump_settings()
	{
		{
			std::vector<string_option> additional_string_options = get_string_options();
			for(std::vector<string_option>::iterator it = additional_string_options.begin(); it != additional_string_options.end(); it++)
				std::cout << it->name << " = " << *it->var << std::endl;
			std::vector<multi_string_option> additional_multi_string_options = get_multi_string_options();
			for(std::vector<multi_string_option>::iterator it = additional_multi_string_options.begin(); it != additional_multi_string_options.end(); it++)
			{
				std::cout << it->name << " = ";
				for(std::vector<std::string>::const_iterator it2 = it->var->begin(); it2 != it->var->end(); ++it2)
				{
					if (it2 != it->var->begin())
						std::cout << ", ";
					std::cout << *it2;
				}
				std::cout << std::endl;
			}
			std::vector<path_option> additional_path_options = get_path_options();
			for(std::vector<path_option>::iterator it = additional_path_options.begin(); it != additional_path_options.end(); it++)
				std::cout << it->name << " = " << *it->var << std::endl;
			std::vector<bool_option> additional_bool_options = get_bool_options();
			for(std::vector<bool_option>::iterator it = additional_bool_options.begin(); it != additional_bool_options.end(); it++)
				std::cout << it->name << " = " << *it->var << std::endl;
			std::vector<float_option> additional_float_options = get_float_options();
			for(std::vector<float_option>::iterator it = additional_float_options.begin(); it != additional_float_options.end(); it++)
				std::cout << it->name << " = " << *it->var << std::endl;
			std::vector<int_option> additional_int_options = get_int_options();
			for(std::vector<int_option>::iterator it = additional_int_options.begin(); it != additional_int_options.end(); it++)
				std::cout << it->name << " = " << *it->var << std::endl;
		}
		{
			std::vector<string_option> additional_string_options = master_factory->get_string_options();
			for(std::vector<string_option>::iterator it = additional_string_options.begin(); it != additional_string_options.end(); it++)
				std::cout << it->name << " = " << *it->var << std::endl;
			std::vector<multi_string_option> additional_multi_string_options = master_factory->get_multi_string_options();
			for(std::vector<multi_string_option>::iterator it = additional_multi_string_options.begin(); it != additional_multi_string_options.end(); it++)
			{
				std::cout << it->name << " = ";
				for(std::vector<std::string>::const_iterator it2 = it->var->begin(); it2 != it->var->end(); ++it2)
				{
					if (it2 != it->var->begin())
						std::cout << ", ";
					std::cout << *it2;
				}
				std::cout << std::endl;
			}
			std::vector<path_option> additional_path_options = master_factory->get_path_options();
			for(std::vector<path_option>::iterator it = additional_path_options.begin(); it != additional_path_options.end(); it++)
				std::cout << it->name << " = " << *it->var << std::endl;
			std::vector<bool_option> additional_bool_options = master_factory->get_bool_options();
			for(std::vector<bool_option>::iterator it = additional_bool_options.begin(); it != additional_bool_options.end(); it++)
				std::cout << it->name << " = " << *it->var << std::endl;
			std::vector<float_option> additional_float_options = master_factory->get_float_options();
			for(std::vector<float_option>::iterator it = additional_float_options.begin(); it != additional_float_options.end(); it++)
				std::cout << it->name << " = " << *it->var << std::endl;
			std::vector<int_option> additional_int_options = master_factory->get_int_options();
			for(std::vector<int_option>::iterator it = additional_int_options.begin(); it != additional_int_options.end(); it++)
				std::cout << it->name << " = " << *it->var << std::endl;
		}
	}

	void toolset::do_custom_action()
	{
		throw std::runtime_error((boost::format("Unknown action: %1%") % action).str());
	}

	network_schema::ptr toolset::load_schema() const
	{
		network_schema::ptr schema(new network_schema());
		{
			boost::filesystem::path filepath = get_working_data_folder() / schema_filename;
			if (!boost::filesystem::exists(filepath))
				throw neural_network_exception((boost::format("Error loading schema, file not found: %1%") % filepath.string()).str());
			boost::filesystem::ifstream in(filepath, std::ios_base::in);
			schema->read_proto(in);
		}
		return schema;
	}

	void toolset::run_inference()
	{
		network_schema::ptr schema = load_schema();
		forward_propagation::ptr forward_prop = forward_propagation_factory->create(*schema, inference_output_layer_names, debug);
		structured_data_bunch_reader::ptr reader = get_reader(inference_dataset_name);

		std::vector<std::pair<unsigned int, boost::filesystem::path> > ann_data_name_and_filepath_list = get_ann_data_index_and_filepath_list();
		for(std::vector<std::pair<unsigned int, boost::filesystem::path> >::const_iterator it = ann_data_name_and_filepath_list.begin(); it != ann_data_name_and_filepath_list.end(); ++it)
		{
			network_data data;
			{
				boost::filesystem::ifstream in(it->second, std::ios_base::in | std::ios_base::binary);
				data.read(in);
			}
			forward_prop->set_data(data);

			neuron_value_set_data_bunch_writer writer;
			forward_propagation::stat st = forward_prop->run(*reader, writer);
			std::cout << "Network # " << it->first << " - " << st << std::endl;

			for(std::map<std::string, std::pair<layer_configuration_specific, neuron_value_set::ptr> >::const_iterator it = writer.layer_name_to_config_and_value_set_map.begin(); it != writer.layer_name_to_config_and_value_set_map.end(); ++it)
				std::cout << schema->get_layer(it->first)->get_string_for_average_data(it->second.first, *it->second.second->get_average()) << std::endl;
		}
	}

	structured_data_bunch_reader::ptr toolset::get_reader(const std::string& dataset_name) const
	{
		throw neural_network_exception((boost::format("get_reader is not implemented for %1% dataset") % dataset_name).str());
	}

	boost::filesystem::path toolset::get_ann_subfolder_name() const
	{
		return ann_subfolder_name;
	}

	std::vector<std::pair<unsigned int, boost::filesystem::path> > toolset::get_ann_data_index_and_filepath_list() const
	{
		std::vector<std::pair<unsigned int, boost::filesystem::path> > res;

		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();

		nnforge_regex expression(trained_ann_index_extractor_pattern);
		nnforge_cmatch what;
		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(batch_folder); it != boost::filesystem::directory_iterator(); ++it)
		{
			boost::filesystem::path file_path = it->path();
			std::string file_name = file_path.filename().string();

			if (nnforge_regex_search(file_name.c_str(), what, expression))
			{
				unsigned int ann_data_index = atol(std::string(what[1].first, what[1].second).c_str());
				if ((inference_ann_data_index != -1) && (inference_ann_data_index != ann_data_index))
					continue;

				res.push_back(std::make_pair(ann_data_index, file_path));
			}
		}

		return res;
	}

	void toolset::dump_schema_dot()
	{
		network_schema::ptr schema = load_schema();

		boost::filesystem::path dot_filename(schema_filename);
		dot_filename.replace_extension("dot");
		boost::filesystem::path filepath = get_working_data_folder() / dot_filename;

		boost::filesystem::ofstream out(filepath, std::ios_base::out);
		schema->write_dot(out);
	}

	void toolset::train()
	{
		network_schema::ptr schema = load_schema();
		network_trainer::ptr trainer = get_network_trainer();

		structured_data_bunch_reader::ptr reader = get_reader(training_dataset_name);

		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();
		boost::filesystem::create_directories(batch_folder);
		boost::filesystem::path batch_resume_folder = batch_folder / ann_resume_subfolder_name;
		boost::filesystem::create_directories(batch_resume_folder);

		std::vector<network_data_peek_entry> leading_tasks;
		if (load_resume)
		{
			leading_tasks = get_resume_ann_list_entry_list();
		}
		unsigned int starting_index = get_starting_index_for_batch_training();
		for(std::vector<network_data_peek_entry>::const_iterator it = leading_tasks.begin(); it != leading_tasks.end(); ++it)
			starting_index = std::max(starting_index, it->index + 1);
		nnforge_shared_ptr<network_data_peeker> peeker = nnforge_shared_ptr<network_data_peeker>(new network_data_peeker_random(ann_count, starting_index, leading_tasks));

		complex_network_data_pusher progress;

		if (dump_resume)
		{
			progress.push_back(network_data_pusher::ptr(new save_resume_network_data_pusher(batch_resume_folder)));
		}

		progress.push_back(network_data_pusher::ptr(new report_progress_network_data_pusher()));

		std::vector<network_data_pusher::ptr> validators_for_training = get_validators_for_training(schema);
		progress.insert(progress.end(), validators_for_training.begin(), validators_for_training.end());

		summarize_network_data_pusher res(batch_folder);

		trainer->train(
			*get_reader(training_dataset_name),
			*peeker,
			progress,
			res);
	}

	std::vector<network_data_pusher::ptr> toolset::get_validators_for_training(network_schema::const_ptr schema)
	{
		std::vector<network_data_pusher::ptr> res;

		if (is_training_with_validation())
		{
			res.push_back(network_data_pusher::ptr(new validate_progress_network_data_pusher(
				forward_propagation_factory->create(*schema, inference_output_layer_names, debug),
				get_reader(inference_dataset_name))));
		}

		return res;
	}

	bool toolset::is_training_with_validation() const
	{
		return true;
	}

	unsigned int toolset::get_starting_index_for_batch_training() const
	{
		nnforge_regex expression(trained_ann_index_extractor_pattern);
		nnforge_cmatch what;

		int max_index = -1;
		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();
		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(batch_folder); it != boost::filesystem::directory_iterator(); it++)
		{
			boost::filesystem::path file_path = it->path();
			std::string file_name = file_path.filename().string();

			if (nnforge_regex_search(file_name.c_str(), what, expression))
			{
				int index = atol(std::string(what[1].first, what[1].second).c_str());
				max_index = std::max<int>(max_index, index); 
			}
		}

		return static_cast<unsigned int>(max_index + 1) + batch_offset;
	}

	std::vector<network_data_peek_entry> toolset::get_resume_ann_list_entry_list() const
	{
		std::vector<network_data_peek_entry> res;

		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();
		boost::filesystem::create_directories(batch_folder);
		boost::filesystem::path resume_ann_folder_path = batch_folder / ann_resume_subfolder_name;
		boost::filesystem::create_directories(resume_ann_folder_path);

		std::set<unsigned int> trained_ann_list = get_trained_ann_list();

		std::map<unsigned int, unsigned int> resume_ann_list = get_resume_ann_list(trained_ann_list);

		network_schema::ptr schema = load_schema();

		for(std::map<unsigned int, unsigned int>::const_iterator it = resume_ann_list.begin(); it != resume_ann_list.end(); ++it)
		{
			network_data_peek_entry new_item;
			new_item.index = it->first;
			new_item.start_epoch = it->second;
			
			{
				std::string filename = (boost::format("ann_trained_%|1$03d|_epoch_%|2$05d|.data") % new_item.index % new_item.start_epoch).str();
				boost::filesystem::path filepath = resume_ann_folder_path / filename;
				new_item.data = network_data::ptr(new network_data());
				boost::filesystem::ifstream in(filepath, std::ios_base::in | std::ios_base::binary);
				new_item.data->read(in);
			}

			{
				std::string momentum_filename = (boost::format("momentum_%|1$03d|.data") % new_item.index).str();
				boost::filesystem::path momentum_filepath = resume_ann_folder_path / momentum_filename;
				if (boost::filesystem::exists(momentum_filepath))
				{
					new_item.momentum_data = network_data::ptr(new network_data());
					boost::filesystem::ifstream in(momentum_filepath, std::ios_base::in | std::ios_base::binary);
					new_item.momentum_data->read(in);
				}
			}

			res.push_back(new_item);
		}

		std::sort(res.begin(), res.end(), compare_entry);

		return res;
	}

	bool toolset::compare_entry(network_data_peek_entry i, network_data_peek_entry j)
	{
		return (i.index > j.index);
	}

	std::map<unsigned int, unsigned int> toolset::get_resume_ann_list(const std::set<unsigned int>& exclusion_ann_list) const
	{
		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();
		boost::filesystem::create_directories(batch_folder);
		boost::filesystem::path resume_ann_folder_path = batch_folder / ann_resume_subfolder_name;
		boost::filesystem::create_directories(resume_ann_folder_path);

		std::map<unsigned int, unsigned int> res;
		nnforge_regex expression(resume_ann_index_extractor_pattern);
		nnforge_cmatch what;

		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(resume_ann_folder_path); it != boost::filesystem::directory_iterator(); ++it)
		{
			if (it->status().type() == boost::filesystem::regular_file)
			{
				boost::filesystem::path file_path = it->path();
				std::string file_name = file_path.filename().string();

				if (nnforge_regex_search(file_name.c_str(), what, expression))
				{
					unsigned int index = static_cast<unsigned int>(atol(std::string(what[1].first, what[1].second).c_str()));
					if (exclusion_ann_list.find(index) == exclusion_ann_list.end())
					{
						unsigned int epoch = static_cast<unsigned int>(atol(std::string(what[2].first, what[2].second).c_str()));
						std::map<unsigned int, unsigned int>::iterator it2 = res.find(index);
						if (it2 == res.end())
							res.insert(std::make_pair(index, epoch));
						else
							it2->second = std::max(it2->second, epoch);
					}
				}
			}
		}

		return res;
	}

	std::set<unsigned int> toolset::get_trained_ann_list() const
	{
		boost::filesystem::path trained_ann_folder_path = get_working_data_folder() / get_ann_subfolder_name();
		boost::filesystem::create_directories(trained_ann_folder_path);

		std::set<unsigned int> res;
		nnforge_regex expression(trained_ann_index_extractor_pattern);
		nnforge_cmatch what;

		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(trained_ann_folder_path); it != boost::filesystem::directory_iterator(); ++it)
		{
			if (it->status().type() == boost::filesystem::regular_file)
			{
				boost::filesystem::path file_path = it->path();
				std::string file_name = file_path.filename().string();

				if (nnforge_regex_search(file_name.c_str(), what, expression))
				{
					unsigned int index = static_cast<unsigned int>(atol(std::string(what[1].first, what[1].second).c_str()));
					res.insert(index);
				}
			}
		}

		return res;
	}

	network_trainer::ptr toolset::get_network_trainer() const
	{
		network_trainer::ptr res;

		network_schema::ptr schema = load_schema();

		backward_propagation::ptr backprop = backward_propagation_factory->create(
			*schema,
			training_output_layer_names,
			training_error_source_layer_names,
			training_exclude_data_update_layer_names,
			debug);

		if (training_algo == "sgd")
		{
			network_trainer_sgd::ptr typed_res(
				new network_trainer_sgd(
					schema,
					training_output_layer_names,
					training_error_source_layer_names,
					training_exclude_data_update_layer_names,
					backprop));

			res = typed_res;
		}
		else
			throw neural_network_exception((boost::format("Unknown training algo specified: %1%") % training_algo).str());

		res->epoch_count = training_epoch_count;
		res->learning_rate = learning_rate;
		res->learning_rate_decay_tail_epoch_count = learning_rate_decay_tail_epoch_count;
		res->learning_rate_decay_rate = learning_rate_decay_rate;
		res->learning_rate_rise_head_epoch_count = learning_rate_rise_head_epoch_count;
		res->learning_rate_rise_rate = learning_rate_rise_rate;
		res->weight_decay = weight_decay;
		res->batch_size = batch_size;
		res->momentum = training_momentum(momentum_type_str, momentum_val);

		return res;
	}
}
