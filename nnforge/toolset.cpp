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

namespace nnforge
{
	const char * toolset::logfile_name = "log.txt";
	const char * toolset::ann_subfolder_name = "batch";
	const char * toolset::debug_subfolder_name = "debug";
	const char * toolset::trained_ann_extractor_pattern = "^ann_trained_(.+)\\.data$";

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

		res.push_back(string_option("action", &action, get_default_action().c_str(), "run action (info, run_inference, dump_schema_dot)"));
		res.push_back(string_option("schema", &schema_filename, "schema.txt", "Name of the file with schema of the network, in protobuf format"));
		res.push_back(string_option("inference_dataset_name", &inference_dataset_name, "validating", "Name of the dataset to be used for inference"));
		res.push_back(string_option("inference_ann_data_name", &inference_ann_data_name, "", "Name of the dataset to be used for inference"));

		return res;
	}

	std::vector<multi_string_option> toolset::get_multi_string_options()
	{
		std::vector<multi_string_option> res;

		res.push_back(multi_string_option("output_layer_name,O", &output_layer_names, "Names of the output layers"));

		return res;
	}

	std::vector<path_option> toolset::get_path_options()
	{
		std::vector<path_option> res;

		res.push_back(path_option("config", &config_file_path, default_config_path.c_str(), "path to the configuration file"));
		res.push_back(path_option("working_data_folder", &working_data_folder, "", "path to the folder where data are processed"));

		return res;
	}

	std::vector<bool_option> toolset::get_bool_options()
	{
		std::vector<bool_option> res;

		res.push_back(bool_option("debug_mode", &debug_mode, false, "debug mode"));

		return res;
	}

	std::vector<float_option> toolset::get_float_options()
	{
		std::vector<float_option> res;

		return res;
	}

	std::vector<int_option> toolset::get_int_options()
	{
		std::vector<int_option> res;

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
		forward_propagation::ptr forward_prop = forward_propagation_factory->create(*schema, output_layer_names, debug);
		structured_data_bunch_reader::ptr reader = get_reader(inference_dataset_name);

		std::vector<std::pair<std::string, boost::filesystem::path> > ann_data_name_and_filepath_list = get_ann_data_name_and_filepath_list();
		for(std::vector<std::pair<std::string, boost::filesystem::path> >::const_iterator it = ann_data_name_and_filepath_list.begin(); it != ann_data_name_and_filepath_list.end(); ++it)
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

	std::vector<std::pair<std::string, boost::filesystem::path> > toolset::get_ann_data_name_and_filepath_list() const
	{
		std::vector<std::pair<std::string, boost::filesystem::path> > res;

		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();

		nnforge_regex expression(trained_ann_extractor_pattern);
		nnforge_cmatch what;
		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(batch_folder); it != boost::filesystem::directory_iterator(); ++it)
		{
			boost::filesystem::path file_path = it->path();
			std::string file_name = file_path.filename().string();

			if (nnforge_regex_search(file_name.c_str(), what, expression))
			{
				std::string data_name = std::string(what[1].first, what[1].second);
				if ((!inference_ann_data_name.empty()) && (inference_ann_data_name != data_name))
					continue;

				res.push_back(std::make_pair(data_name, file_path));
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
}
