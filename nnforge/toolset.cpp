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
#include "save_snapshot_network_data_pusher.h"
#include "report_progress_network_data_pusher.h"
#include "summarize_network_data_pusher.h"
#include "validate_progress_network_data_pusher.h"
#include "structured_data_stream_writer.h"
#include "structured_data_bunch_stream_reader.h"
#include "data_visualizer.h"
#include "transformed_structured_data_reader.h"

namespace nnforge
{
	const char * toolset::logfile_name = "log.txt";
	const char * toolset::ann_subfolder_name = "trained_data";
	const char * toolset::debug_subfolder_name = "debug";
	const char * toolset::dump_data_subfolder_name = "dump_data";
	const char * toolset::trained_ann_index_extractor_pattern = "^ann_trained_(\\d+)$";
	const char * toolset::snapshot_ann_index_extractor_pattern = "^ann_trained_(\\d+)_epoch_(\\d+)$";
	const char * toolset::ann_snapshot_subfolder_name = "snapshots";
	const char * toolset::dataset_extractor_pattern = "^%1%_(.+)\\.dt$";

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
		else if (!action.compare("dump_schema"))
		{
			dump_schema_gv();
		}
		else if (!action.compare("train"))
		{
			train();
		}
		else if (!action.compare("prepare_training_data"))
		{
			prepare_training_data();
		}
		else if (!action.compare("prepare_testing_data"))
		{
			prepare_testing_data();
		}
		else if (!action.compare("shuffle_data"))
		{
			shuffle_data();
		}
		else if (!action.compare("dump_data"))
		{
			dump_data();
		}
		else if (!action.compare("create_normalizer"))
		{
			create_normalizer();
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

		forward_prop_factory = master_factory->create_forward_propagation_factory();
		backward_prop_factory = master_factory->create_backward_propagation_factory();

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

		res.push_back(string_option("action", &action, get_default_action().c_str(), "run action (info, prepare_training_data, prepare_testing_data, shuffle_data, dump_data, dump_schema, create_normalizer, run_inference, train)"));
		res.push_back(string_option("schema", &schema_filename, "schema.txt", "Name of the file with schema of the network, in protobuf format"));
		res.push_back(string_option("inference_dataset_name", &inference_dataset_name, "validating", "Name of the dataset to be used for inference"));
		res.push_back(string_option("training_dataset_name", &training_dataset_name, "training", "Name of the dataset to be used for training"));
		res.push_back(string_option("shuffle_dataset_name", &shuffle_dataset_name, "training", "Name of the dataset to be shuffled"));
		res.push_back(string_option("training_algo", &training_algo, "", "Training algorithm (sgd)"));
		res.push_back(string_option("momentum_type", &momentum_type_str, "vanilla", "Type of the momentum to use (none, vanilla, nesterov)"));
		res.push_back(string_option("inference_mode", &inference_mode, "report_average_per_entry", "What to do with inference_output_layer_name (report_average_per_nn, dump_average_across_nets)"));
		res.push_back(string_option("inference_output_dataset_name", &inference_output_dataset_name, "", "Name of the dataset dumped during inference, empty value means using inference_dataset_name"));
		res.push_back(string_option("dump_dataset_name", &dump_dataset_name, "training", "Name of the dataset to dump data from"));
		res.push_back(string_option("dump_layer_name", &dump_layer_name, "", "Name of the layer to dump data from"));
		res.push_back(string_option("dump_extension_image", &dump_extension_image, "jpg", "Extension (type) of the files for dumping 2D data"));
		res.push_back(string_option("dump_extension_video", &dump_extension_video, "avi", "Extension (type) of the files for dumping 3D data"));
		res.push_back(string_option("normalizer_dataset_name", &normalizer_dataset_name, "training", "Name of the dataset to create normalizer from"));
		res.push_back(string_option("normalizer_layer_name", &normalizer_layer_name, "", "Name of the layer to create normalizer for"));

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
		res.push_back(path_option("input_data_folder", &input_data_folder, "", "Path to the folder where input data are located"));

		return res;
	}

	std::vector<bool_option> toolset::get_bool_options()
	{
		std::vector<bool_option> res;

		res.push_back(bool_option("debug_mode", &debug_mode, false, "Debug mode"));
		res.push_back(bool_option("resume_from_snapshot,R", &resume_from_snapshot, false, "Continue neural network training starting from saved snapshot"));
		res.push_back(bool_option("dump_snapshot", &dump_snapshot, true, "Dump neural network data after each epoch"));
		res.push_back(bool_option("dump_data_rgb", &dump_data_rgb, true, "Treat 3 feature map data layer as RGB"));

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
		res.push_back(int_option("dump_data_sample_count", &dump_data_sample_count, 100, "Samples to dump"));
		res.push_back(int_option("dump_data_scale", &dump_data_scale, 1, "Scale dumped data dimensions by this value"));
		res.push_back(int_option("dump_data_video_fps", &dump_data_video_fps, 5, "Frames per second when dumping videos"));
		res.push_back(int_option("epoch_count_in_training_dataset", &epoch_count_in_training_dataset, 1, "The whole training dataset should be split in this amount of epochs"));
		res.push_back(int_option("dump_compact_samples", &dump_compact_samples, 1, "Compact (average) results acrioss samples for inference of type dump_average_across_nets"));

		return res;
	}

	boost::filesystem::path toolset::get_working_data_folder() const
	{
		return working_data_folder;
	}

	boost::filesystem::path toolset::get_input_data_folder() const
	{
		return input_data_folder;
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
		forward_propagation::ptr forward_prop = forward_prop_factory->create(*schema, inference_output_layer_names, debug);
		structured_data_bunch_reader::ptr reader = get_structured_data_bunch_reader(inference_dataset_name, dataset_usage_inference, 1);

		std::vector<std::pair<unsigned int, boost::filesystem::path> > ann_data_name_and_folderpath_list = get_ann_data_index_and_folderpath_list();
		std::map<std::string, std::pair<layer_configuration_specific, neuron_value_set::ptr> > average_layer_name_to_config_and_value_set_map;
		unsigned int accumulated_count = 0;

		if (forward_prop->is_schema_with_weights())
		{
			for(std::vector<std::pair<unsigned int, boost::filesystem::path> >::const_iterator it = ann_data_name_and_folderpath_list.begin(); it != ann_data_name_and_folderpath_list.end(); ++it)
			{
				network_data data;
				data.read(it->second);
				forward_prop->set_data(data);

				neuron_value_set_data_bunch_writer writer;
				forward_propagation::stat st = forward_prop->run(*reader, writer);
				std::cout << "NN # " << it->first << " - " << st << std::endl;

				for(std::map<std::string, std::pair<layer_configuration_specific, neuron_value_set::ptr> >::iterator it2 = writer.layer_name_to_config_and_value_set_map.begin(); it2 != writer.layer_name_to_config_and_value_set_map.end(); ++it2)
				{
					if (inference_mode == "report_average_per_entry")
						std::cout << schema->get_layer(it2->first)->get_string_for_average_data(it2->second.first, *it2->second.second->get_average()) << std::endl;
					else if (inference_mode == "dump_average_across_nets")
					{
						it2->second.second->compact(dump_compact_samples);

						if (it == ann_data_name_and_folderpath_list.begin())
							average_layer_name_to_config_and_value_set_map.insert(*it2);
						else
						{
							float alpha = 1.0F / static_cast<float>(accumulated_count + 1);
							float beta = 1.0F - alpha;
							average_layer_name_to_config_and_value_set_map[it2->first].second->add(*it2->second.second, alpha, beta);
						}
					}
					else
						throw neural_network_exception((boost::format("Unknown inference_mode specified: %1%") % inference_mode).str());
				}

				++accumulated_count;
			}
		}
		else
		{
			network_data data;
			forward_prop->set_data(data);

			neuron_value_set_data_bunch_writer writer;
			forward_propagation::stat st = forward_prop->run(*reader, writer);
			std::cout << "NN <no weights uniform> - " << st << std::endl;

			for(std::map<std::string, std::pair<layer_configuration_specific, neuron_value_set::ptr> >::const_iterator it2 = writer.layer_name_to_config_and_value_set_map.begin(); it2 != writer.layer_name_to_config_and_value_set_map.end(); ++it2)
			{
				if (inference_mode == "report_average_per_entry")
					std::cout << schema->get_layer(it2->first)->get_string_for_average_data(it2->second.first, *it2->second.second->get_average()) << std::endl;
				else if (inference_mode == "dump_average_across_nets")
				{
					average_layer_name_to_config_and_value_set_map.insert(*it2);
				}
				else
					throw neural_network_exception((boost::format("Unknown inference_mode specified: %1%") % inference_mode).str());
			}

			++accumulated_count;
		}

		if (inference_mode == "dump_average_across_nets")
		{
			for(std::map<std::string, std::pair<layer_configuration_specific, neuron_value_set::ptr> >::const_iterator it = average_layer_name_to_config_and_value_set_map.begin(); it != average_layer_name_to_config_and_value_set_map.end(); ++it)
			{
				std::string dataset_name = inference_output_dataset_name.empty() ? inference_dataset_name : inference_output_dataset_name;
				std::string file_name = (boost::format("%1%_%2%.dt") % dataset_name % it->first).str();
				boost::filesystem::path file_path = get_working_data_folder() / file_name;
				std::cout << "Writing " << file_path.string() << std::endl;
				nnforge_shared_ptr<std::ostream> out(new boost::filesystem::ofstream(file_path, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary));
				{
					structured_data_stream_writer dw(out, it->second.first);
					const std::vector<nnforge_shared_ptr<std::vector<float> > >& data = it->second.second->neuron_value_list;
					for(std::vector<nnforge_shared_ptr<std::vector<float> > >::const_iterator data_it = data.begin(); data_it != data.end(); ++data_it)
					{
						const std::vector<float>& dd = **data_it;
						dw.write(&dd[0]);
					}
				}
			}
		}
	}

	structured_data_bunch_reader::ptr toolset::get_structured_data_bunch_reader(
		const std::string& dataset_name,
		dataset_usage usage,
		unsigned int multiple_epoch_count) const
	{
		std::map<std::string, boost::filesystem::path> data_filenames = get_data_filenames(dataset_name);

		std::map<std::string, structured_data_reader::ptr> data_reader_map;
		for(std::map<std::string, boost::filesystem::path>::const_iterator it = data_filenames.begin(); it != data_filenames.end(); ++it)
		{
			nnforge_shared_ptr<std::istream> in(new boost::filesystem::ifstream(it->second, std::ios_base::in | std::ios_base::binary));
			structured_data_reader::ptr dr = apply_transformers(get_structured_reader(dataset_name, it->first, in), get_data_transformer_list(dataset_name, it->first, usage));
			data_reader_map.insert(std::make_pair(it->first, dr));
		}

		structured_data_bunch_reader::ptr res(new structured_data_bunch_stream_reader(data_reader_map, multiple_epoch_count));
		return res;
	}

	boost::filesystem::path toolset::get_ann_subfolder_name() const
	{
		return ann_subfolder_name;
	}

	std::vector<std::pair<unsigned int, boost::filesystem::path> > toolset::get_ann_data_index_and_folderpath_list() const
	{
		std::vector<std::pair<unsigned int, boost::filesystem::path> > res;

		boost::filesystem::path trained_data_folder = get_working_data_folder() / get_ann_subfolder_name();

		nnforge_regex expression(trained_ann_index_extractor_pattern);
		nnforge_cmatch what;
		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(trained_data_folder); it != boost::filesystem::directory_iterator(); ++it)
		{
			boost::filesystem::path folder_path = it->path();
			std::string folder_name = folder_path.filename().string();

			if (nnforge_regex_search(folder_name.c_str(), what, expression))
			{
				unsigned int ann_data_index = atol(std::string(what[1].first, what[1].second).c_str());
				if ((inference_ann_data_index != -1) && (inference_ann_data_index != ann_data_index))
					continue;

				res.push_back(std::make_pair(ann_data_index, folder_path));
			}
		}

		return res;
	}

	void toolset::dump_schema_gv()
	{
		network_schema::ptr schema = load_schema();

		boost::filesystem::path gv_filename(schema_filename);
		gv_filename.replace_extension("gv");
		boost::filesystem::path filepath = get_working_data_folder() / gv_filename;

		boost::filesystem::ofstream out(filepath, std::ios_base::out);
		schema->write_gv(out);
	}

	void toolset::train()
	{
		network_schema::ptr schema = load_schema();
		network_trainer::ptr trainer = get_network_trainer();

		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();
		boost::filesystem::create_directories(batch_folder);
		boost::filesystem::path batch_snapshot_folder = batch_folder / ann_snapshot_subfolder_name;
		boost::filesystem::create_directories(batch_snapshot_folder);

		std::vector<network_data_peek_entry> leading_tasks;
		if (resume_from_snapshot)
		{
			leading_tasks = get_snapshot_ann_list_entry_list();
		}
		unsigned int starting_index = get_starting_index_for_batch_training();
		for(std::vector<network_data_peek_entry>::const_iterator it = leading_tasks.begin(); it != leading_tasks.end(); ++it)
			starting_index = std::max(starting_index, it->index + 1);
		nnforge_shared_ptr<network_data_peeker> peeker = nnforge_shared_ptr<network_data_peeker>(new network_data_peeker_random(ann_count, starting_index, leading_tasks));

		complex_network_data_pusher progress;

		if (dump_snapshot)
		{
			progress.push_back(network_data_pusher::ptr(new save_snapshot_network_data_pusher(batch_snapshot_folder)));
		}

		progress.push_back(network_data_pusher::ptr(new report_progress_network_data_pusher()));

		std::vector<network_data_pusher::ptr> validators_for_training = get_validators_for_training(schema);
		progress.insert(progress.end(), validators_for_training.begin(), validators_for_training.end());

		summarize_network_data_pusher res(batch_folder);

		structured_data_bunch_reader::ptr reader = get_structured_data_bunch_reader(training_dataset_name, dataset_usage_train, epoch_count_in_training_dataset);
		trainer->train(
			*reader,
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
				forward_prop_factory->create(*schema, inference_output_layer_names, debug),
				get_structured_data_bunch_reader(inference_dataset_name, dataset_usage_validate_when_train, 1))));
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
			boost::filesystem::path folder_path = it->path();
			std::string folder_name = folder_path.filename().string();

			if (nnforge_regex_search(folder_name.c_str(), what, expression))
			{
				int index = atol(std::string(what[1].first, what[1].second).c_str());
				max_index = std::max<int>(max_index, index); 
			}
		}

		return static_cast<unsigned int>(max_index + 1) + batch_offset;
	}

	std::vector<network_data_peek_entry> toolset::get_snapshot_ann_list_entry_list() const
	{
		std::vector<network_data_peek_entry> res;

		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();
		boost::filesystem::create_directories(batch_folder);
		boost::filesystem::path snapshot_ann_folder_path = batch_folder / ann_snapshot_subfolder_name;
		boost::filesystem::create_directories(snapshot_ann_folder_path);

		std::set<unsigned int> trained_ann_list = get_trained_ann_list();

		std::map<unsigned int, unsigned int> snapshot_ann_list = get_snapshot_ann_list(trained_ann_list);

		network_schema::ptr schema = load_schema();

		for(std::map<unsigned int, unsigned int>::const_iterator it = snapshot_ann_list.begin(); it != snapshot_ann_list.end(); ++it)
		{
			network_data_peek_entry new_item;
			new_item.index = it->first;
			new_item.start_epoch = it->second;
			
			{
				std::string folder_name = (boost::format("ann_trained_%|1$03d|_epoch_%|2$05d|") % new_item.index % new_item.start_epoch).str();
				boost::filesystem::path folder_path = snapshot_ann_folder_path / folder_name;
				new_item.data = network_data::ptr(new network_data());
				new_item.data->read(folder_path);
			}

			{
				std::string momentum_folder_name = (boost::format("momentum_%|1$03d|") % new_item.index).str();
				boost::filesystem::path momentum_folder_path = snapshot_ann_folder_path / momentum_folder_name;
				if (boost::filesystem::exists(momentum_folder_path))
				{
					new_item.momentum_data = network_data::ptr(new network_data());
					new_item.momentum_data->read(momentum_folder_path);
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

	std::map<unsigned int, unsigned int> toolset::get_snapshot_ann_list(const std::set<unsigned int>& exclusion_ann_list) const
	{
		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();
		boost::filesystem::create_directories(batch_folder);
		boost::filesystem::path snapshot_ann_folder_path = batch_folder / ann_snapshot_subfolder_name;
		boost::filesystem::create_directories(snapshot_ann_folder_path);

		std::map<unsigned int, unsigned int> res;
		nnforge_regex expression(snapshot_ann_index_extractor_pattern);
		nnforge_cmatch what;

		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(snapshot_ann_folder_path); it != boost::filesystem::directory_iterator(); ++it)
		{
			if (it->status().type() == boost::filesystem::directory_file)
			{
				boost::filesystem::path folder_path = it->path();
				std::string folder_name = folder_path.filename().string();

				if (nnforge_regex_search(folder_name.c_str(), what, expression))
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
			if (it->status().type() == boost::filesystem::directory_file)
			{
				boost::filesystem::path folder_path = it->path();
				std::string folder_name = folder_path.filename().string();

				if (nnforge_regex_search(folder_name.c_str(), what, expression))
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

		backward_propagation::ptr backprop = backward_prop_factory->create(
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

	void toolset::prepare_testing_data()
	{
		throw std::runtime_error("This toolset doesn't implement preparing testing data");
	}
	
	void toolset::prepare_training_data()
	{
		throw std::runtime_error("This toolset doesn't implement preparing training data");
	}

	void toolset::shuffle_data()
	{
		std::map<std::string, boost::filesystem::path> data_filenames = get_data_filenames(shuffle_dataset_name);

		int entry_count = -1;
		for(std::map<std::string, boost::filesystem::path>::const_iterator it = data_filenames.begin(); it != data_filenames.end(); ++it)
		{
			nnforge_shared_ptr<std::istream> in(new boost::filesystem::ifstream(it->second, std::ios_base::in | std::ios_base::binary));
			structured_data_stream_reader dr(in);
			int new_entry_count = dr.get_entry_count();
			if (new_entry_count < 0)
				throw std::runtime_error((boost::format("Unknown entry count in %1%") % it->second.string()).str());
			if (entry_count < 0)
				entry_count = new_entry_count;
			else if (entry_count != new_entry_count)
				throw std::runtime_error((boost::format("Entry count mismatch: %1% and %2%") % entry_count % new_entry_count).str());
		}
		if (entry_count < 0)
			throw std::runtime_error((boost::format("No data found for dataset %1%") % shuffle_dataset_name).str());
		else if (entry_count == 0)
		{
			std::cout << (boost::format("No data found for dataset %1%") % shuffle_dataset_name).str() << std::endl;
			return;
		}

		std::cout << "Shuffling " << entry_count << " entries in " << shuffle_dataset_name << " dataset" << std::endl;

		std::vector<unsigned int> shuffled_indexes(entry_count);
		{
			for(unsigned int i = 0; i < static_cast<unsigned int>(entry_count); ++i)
				shuffled_indexes[i] = i;
			random_generator rnd = rnd::get_random_generator();
			for(unsigned int i = static_cast<unsigned int>(entry_count) - 1; i > 0; --i)
			{
				nnforge_uniform_int_distribution<unsigned int> dist(0, i);
				unsigned int index = dist(rnd);
				std::swap(shuffled_indexes[i], shuffled_indexes[index]);
			}
		}

		for(std::map<std::string, boost::filesystem::path>::const_iterator it = data_filenames.begin(); it != data_filenames.end(); ++it)
		{
			const boost::filesystem::path& file_path = it->second;
			boost::filesystem::path temp_file_path = file_path;
			temp_file_path += ".tmp";
			{
				std::cout << "Shuffling from " << file_path.string() << " to " << temp_file_path.string() << std::endl;
				nnforge_shared_ptr<std::istream> in(new boost::filesystem::ifstream(file_path, std::ios_base::in | std::ios_base::binary));
				nnforge_shared_ptr<std::ostream> out(new boost::filesystem::ofstream(temp_file_path, std::ios_base::out | std::ios_base::trunc | std::ios_base::binary));
				{
					raw_data_reader::ptr dr = get_raw_reader(shuffle_dataset_name, it->first, in);
					raw_data_writer::ptr dw = dr->get_writer(out);
					std::vector<unsigned char> dt;
					for(unsigned int i = 0; i < static_cast<unsigned int>(entry_count); ++i)
					{
						dr->raw_read(shuffled_indexes[i], dt);
						dw->raw_write(&dt[0], dt.size());
					}
				}
			}
			std::cout << "Renaming " << temp_file_path.string() << " to " << file_path.string() << std::endl;
			boost::filesystem::rename(temp_file_path, file_path);
		}
	}

	raw_data_reader::ptr toolset::get_raw_reader(
		const std::string& dataset_name,
		const std::string& layer_name,
		nnforge_shared_ptr<std::istream> in) const
	{
		return get_structured_reader(dataset_name, layer_name, in);
	}

	structured_data_reader::ptr toolset::get_structured_reader(
		const std::string& dataset_name,
		const std::string& layer_name,
		nnforge_shared_ptr<std::istream> in) const
	{
		return structured_data_reader::ptr(new structured_data_stream_reader(in));
	}

	std::map<std::string, boost::filesystem::path> toolset::get_data_filenames(const std::string& dataset_name) const
	{
		boost::filesystem::path folder_path = get_working_data_folder();

		std::map<std::string, boost::filesystem::path> res;
		nnforge_regex expression((boost::format(dataset_extractor_pattern) % dataset_name).str());
		nnforge_cmatch what;

		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(folder_path); it != boost::filesystem::directory_iterator(); ++it)
		{
			if (it->status().type() == boost::filesystem::regular_file)
			{
				boost::filesystem::path file_path = it->path();
				std::string file_name = file_path.filename().string();

				if (nnforge_regex_search(file_name.c_str(), what, expression))
				{
					std::string data_name = std::string(what[1].first, what[1].second);
					res.insert(std::make_pair(data_name, file_path));
				}
			}
		}

		return res;
	}

	void toolset::dump_data()
	{
		boost::filesystem::path dump_data_folder = get_working_data_folder() / dump_data_subfolder_name;
		std::cout << "Dumping up to " << dump_data_sample_count << " samples to " << dump_data_folder.string() << std::endl;
		boost::filesystem::create_directories(dump_data_folder);

		std::string file_name = (boost::format("%1%_%2%.dt") % dump_dataset_name % dump_layer_name).str();
		boost::filesystem::path file_path = get_working_data_folder() / file_name;
		if (!boost::filesystem::exists(file_path))
			throw neural_network_exception((boost::format("File %1% doesn't exist") % file_path.string()).str());
		nnforge_shared_ptr<std::istream> in(new boost::filesystem::ifstream(file_path, std::ios_base::in | std::ios_base::binary));
		structured_data_reader::ptr dr = apply_transformers(get_structured_reader(dump_dataset_name, dump_layer_name, in), get_data_transformer_list(dump_dataset_name, dump_layer_name, dataset_usage_dump_data));

		layer_configuration_specific config = dr->get_configuration();
		std::vector<unsigned int> dump_data_dimension_list = get_dump_data_dimension_list(static_cast<unsigned int>(config.dimension_sizes.size()));
		std::vector<float> dt(config.get_neuron_count());
		for(int sample_id = 0; sample_id < dump_data_sample_count; ++sample_id)
		{
			if (!dr->read(sample_id, &dt[0]))
				break;

			if (config.dimension_sizes.size() == 2)
			{
				boost::filesystem::path dump_file_path = dump_data_folder / (boost::format("%1%_%2%_%|3$05d|.%4%") % dump_dataset_name % dump_layer_name % sample_id % dump_extension_image).str();

				data_visualizer::save_2d(
					config,
					&dt[0],
					dump_file_path.string().c_str(),
					dump_data_rgb && (config.feature_map_count == 3),
					dump_data_scale,
					dump_data_dimension_list);
			}
			else if (config.dimension_sizes.size() == 3)
			{
				boost::filesystem::path dump_file_path = dump_data_folder / (boost::format("%1%_%2%_%|3$05d|.%4%") % dump_dataset_name % dump_layer_name % sample_id % dump_extension_video).str();

				data_visualizer::save_3d(
					config,
					&dt[0],
					dump_file_path.string().c_str(),
					dump_data_rgb && (config.feature_map_count == 3),
					dump_data_video_fps,
					dump_data_scale,
					dump_data_dimension_list);
			}
			else
				throw neural_network_exception((boost::format("Saving snapshot for %1% dimensions is not implemented") % config.dimension_sizes.size()).str());
		}
	}

	std::vector<unsigned int> toolset::get_dump_data_dimension_list(unsigned int original_dimension_count) const
	{
		std::vector<unsigned int> res;
		
		for(unsigned int i = 0; i < original_dimension_count; ++i)
			res.push_back(i);

		return res;
	}

	std::vector<data_transformer::ptr> toolset::get_data_transformer_list(
		const std::string& dataset_name,
		const std::string& layer_name,
		dataset_usage usage) const
	{
		return std::vector<data_transformer::ptr>();
	}

	structured_data_reader::ptr toolset::apply_transformers(
		structured_data_reader::ptr original_reader,
		const std::vector<data_transformer::ptr>& data_transformer_list) const
	{
		structured_data_reader::ptr current_reader = original_reader;
		for(std::vector<data_transformer::ptr>::const_iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
		{
			structured_data_reader::ptr new_reader(new transformed_structured_data_reader(current_reader, *it));
			current_reader = new_reader;
		}
		return current_reader;
	}

	void toolset::create_normalizer()
	{
		std::string reader_file_name = (boost::format("%1%_%2%.dt") % normalizer_dataset_name % normalizer_layer_name).str();
		boost::filesystem::path reader_file_path = get_working_data_folder() / reader_file_name;

		std::string normalizer_file_name = (boost::format("normalizer_%1%.txt") % normalizer_layer_name).str();
		boost::filesystem::path normalizer_file_path = get_working_data_folder() / normalizer_file_name;

		std::cout << "Generating normalizer file " << normalizer_file_path.string() << " from " << reader_file_path.string() << std::endl;

		if (!boost::filesystem::exists(reader_file_path))
			throw neural_network_exception((boost::format("File %1% doesn't exist") % reader_file_path.string()).str());
		nnforge_shared_ptr<std::istream> in(new boost::filesystem::ifstream(reader_file_path, std::ios_base::in | std::ios_base::binary));
		structured_data_reader::ptr reader = apply_transformers(get_structured_reader(normalizer_dataset_name, normalizer_layer_name, in), get_data_transformer_list(normalizer_dataset_name, normalizer_layer_name, dataset_usage_create_normalizer));

		std::vector<nnforge::feature_map_data_stat> feature_map_data_stat_list = reader->get_feature_map_data_stat_list();
		unsigned int feature_map_id = 0;
		for(std::vector<nnforge::feature_map_data_stat>::const_iterator it = feature_map_data_stat_list.begin(); it != feature_map_data_stat_list.end(); ++it, ++feature_map_id)
			std::cout << "Feature map # " << feature_map_id << ": " << *it << std::endl;

		normalize_data_transformer normalizer(feature_map_data_stat_list);
		boost::filesystem::ofstream file_with_schema(get_working_data_folder() / normalizer_file_name, std::ios_base::out | std::ios_base::trunc);
		normalizer.write_proto(file_with_schema);
	}

	normalize_data_transformer::ptr toolset::get_normalize_data_transformer(const std::string& layer_name) const
	{
		std::string normalizer_file_name = (boost::format("normalizer_%1%.txt") % layer_name).str();
		boost::filesystem::path normalizer_file_path = get_working_data_folder() / normalizer_file_name;

		if (!boost::filesystem::exists(normalizer_file_path))
			return normalize_data_transformer::ptr();

		boost::filesystem::ifstream file_with_schema(get_working_data_folder() / normalizer_file_name, std::ios_base::in);
		normalize_data_transformer::ptr res(new normalize_data_transformer());
		res->read_proto(file_with_schema);

		return res;
	}
}
