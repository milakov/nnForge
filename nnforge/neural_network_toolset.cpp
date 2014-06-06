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

#include "neural_network_toolset.h"

#include <boost/program_options.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/format.hpp>
#include <boost/chrono.hpp>

#include <regex>
#include <algorithm>
#include <numeric>

#include "snapshot_visualizer.h"
#include "output_neuron_class_set.h"
#include "classifier_result.h"
#include "supervised_data_stream_reader.h"
#include "unsupervised_data_stream_reader.h"
#include "validate_progress_network_data_pusher.h"
#include "network_data_peeker.h"
#include "network_data_peeker_random.h"
#include "network_data_peeker_single.h"
#include "network_data_pusher.h"
#include "report_progress_network_data_pusher.h"
#include "complex_network_data_pusher.h"
#include "testing_complete_result_set_classifier_visualizer.h"
#include "testing_complete_result_set_roc_visualizer.h"
#include "network_trainer_sdlm.h"
#include "summarize_network_data_pusher.h"
#include "supervised_transformed_input_data_reader.h"
#include "supervised_transformed_output_data_reader.h"
#include "normalize_data_transformer.h"
#include "unsupervised_transformed_input_data_reader.h"
#include "mse_error_function.h"
#include "nn_types.h"
#include "supervised_data_stream_writer.h"
#include "supervised_multiple_epoch_data_reader.h"
#include "supervised_limited_entry_count_data_reader.h"
#include "network_trainer_sgd.h"
#include "save_resume_network_data_pusher.h"
#include "network_data_peeker_load_resume.h"

namespace nnforge
{
	const char * neural_network_toolset::training_data_filename = "training.sdt";
	const char * neural_network_toolset::training_randomized_data_filename = "training_randomized.sdt";
	const char * neural_network_toolset::validating_data_filename = "validating.sdt";
	const char * neural_network_toolset::testing_data_filename = "testing.sdt";
	const char * neural_network_toolset::testing_unsupervised_data_filename = "testing.udt";
	const char * neural_network_toolset::schema_filename = "ann.schema";
	const char * neural_network_toolset::normalizer_input_filename = "normalizer_input.data";
	const char * neural_network_toolset::normalizer_output_filename = "normalizer_output.data";
	const char * neural_network_toolset::snapshot_subfolder_name = "snapshot";
	const char * neural_network_toolset::snapshot_data_subfolder_name = "snapshot_data";
	const char * neural_network_toolset::ann_snapshot_subfolder_name = "ann_snapshot";
	const char * neural_network_toolset::snapshot_invalid_subfolder_name = "invalid";
	const char * neural_network_toolset::ann_subfolder_name = "batch";
	const char * neural_network_toolset::ann_resume_subfolder_name = "resume";
	const char * neural_network_toolset::trained_ann_index_extractor_pattern = "^ann_trained_(\\d+)\\.data$";
	const char * neural_network_toolset::logfile_name = "log.txt";

	neural_network_toolset::neural_network_toolset(factory_generator_smart_ptr factory)
		: factory(factory)
	{
	}

	neural_network_toolset::~neural_network_toolset()
	{
	}

	void neural_network_toolset::do_action()
	{
		if (!action.compare("create"))
		{
			create();
		}
		else if (!action.compare("prepare_training_data"))
		{
			prepare_training_data();
		}
		else if (!action.compare("prepare_testing_data"))
		{
			prepare_testing_data();
		}
		else if (!action.compare("randomize_data"))
		{
			randomize_data();
		}
		else if (!action.compare("generate_input_normalizer"))
		{
			generate_input_normalizer();
		}
		else if (!action.compare("generate_output_normalizer"))
		{
			generate_output_normalizer();
		}
		else if (!action.compare("validate"))
		{
			validate(true);
		}
		else if (!action.compare("test"))
		{
			validate(false);
		}
		else if (!action.compare("info"))
		{
			factory->info();
		}
		else if (!action.compare("train"))
		{
			train();
		}
		else if (!action.compare("profile_updater"))
		{
			profile_updater();
		}
		else if (!action.compare("profile_hessian"))
		{
			profile_hessian();
		}
		else if (!action.compare("snapshot"))
		{
			snapshot();
		}
		else if (!action.compare("snapshot_data"))
		{
			snapshot_data();
		}
		else if (!action.compare("snapshot_invalid"))
		{
			snapshot_invalid();
		}
		else if (!action.compare("ann_snapshot"))
		{
			ann_snapshot();
		}
		else
		{
			throw std::runtime_error((boost::format("Unknown action: %1%") % action).str());
		}
	}

	void neural_network_toolset::prepare_testing_data()
	{
		throw std::runtime_error("This toolset doesn't implement preparing testing data");
	}
	
	bool neural_network_toolset::parse(int argc, char* argv[])
	{
		boost::filesystem::path config_file;

		std::string default_config_path = argv[0];
		default_config_path += ".cfg";

		// Declare a group of options that will be 
		// allowed only on command line
		boost::program_options::options_description gener("Generic options");
		gener.add_options()
			("help", "produce help message")
			("action,A", boost::program_options::value<std::string>(&action), "run action (info, create, prepare_training_data, prepare_testing_data, randomize_data, generate_input_normalizer, generate_output_normalizer, test, test_batch, validate, validate_batch, validate_infinite, train, snapshot, snapshot_invalid, ann_snapshot, profile_updater, profile_hessian)")
			("config,C", boost::program_options::value<boost::filesystem::path>(&config_file)->default_value(default_config_path), "path to the configuration file.")
			;

		// Declare a group of options that will be 
		// allowed both on command line and in
		// config file
		boost::program_options::options_description config("Configuration");
		config.add_options()
			("input_data_folder,I", boost::program_options::value<boost::filesystem::path>(&input_data_folder)->default_value(""), "path to the folder where input data are located.")
			("working_data_folder,W", boost::program_options::value<boost::filesystem::path>(&working_data_folder)->default_value(""), "path to the folder where data are processed.")
			("ann_count,N", boost::program_options::value<unsigned int>(&ann_count)->default_value(1), "amount of networks to train.")
			("training_epoch_count,E", boost::program_options::value<unsigned int>(&training_epoch_count)->default_value(50), "amount of epochs to perform during single ANN training.")
			("snapshot_count", boost::program_options::value<unsigned int>(&snapshot_count)->default_value(100), "amount of snapshots to generate.")
			("snapshot_extension", boost::program_options::value<std::string>(&snapshot_extension)->default_value("jpg"), "Extension (type) of the files for neuron values snapshots.")
			("snapshot_mode", boost::program_options::value<std::string>(&snapshot_mode)->default_value("image"), "Type of the neuron values snapshot to generate (image, video).")
			("snapshot_video_fps", boost::program_options::value<unsigned int>(&snapshot_video_fps)->default_value(5), "Frames per second when saving video snapshot.")
			("snapshot_ann_index", boost::program_options::value<unsigned int>(&snapshot_ann_index)->default_value(0), "Index of ANN for snapshots.")
			("mu_increase_factor", boost::program_options::value<float>(&mu_increase_factor)->default_value(1.0F), "Mu increases by this ratio each epoch.")
			("max_mu", boost::program_options::value<float>(&max_mu)->default_value(1.0F), "Maximum Mu during training.")
			("per_layer_mu", boost::program_options::value<bool>(&per_layer_mu)->default_value(false), "Mu is calculated for each layer separately.")
			("learning_rate,L", boost::program_options::value<float>(&learning_rate)->default_value(0.02F), "Global learning rate, Eta/Mu ratio for Stochastic Diagonal Levenberg Marquardt.")
			("learning_rate_decay_tail", boost::program_options::value<unsigned int>(&learning_rate_decay_tail_epoch_count)->default_value(0), "Number of tail iterations with gradually lowering learning rates.")
			("learning_rate_decay_rate", boost::program_options::value<float>(&learning_rate_decay_rate)->default_value(0.5F), "Degradation of learning rate at each tail epoch.")
			("learning_rate_rise_head", boost::program_options::value<unsigned int>(&learning_rate_rise_head_epoch_count)->default_value(0), "Number of head iterations with gradually increasing learning rates.")
			("learning_rate_rise_rate", boost::program_options::value<float>(&learning_rate_rise_rate)->default_value(0.1F), "Increase factor of learning rate at each head epoch (<1.0).")
			("batch_offset", boost::program_options::value<unsigned int>(&batch_offset)->default_value(0), "shift initial ANN ID when batch training.")
			("test_validate_ann_index", boost::program_options::value<int>(&test_validate_ann_index)->default_value(-1), "Index of ANN to test/validate. -1 indicates all ANNs, batch mode.")
			("snapshot_data_set", boost::program_options::value<std::string>(&snapshot_data_set)->default_value("training"), "Type of the dataset to use for snapshots (training, validating, testing).")
			("profile_updater_entry_count", boost::program_options::value<unsigned int>(&profile_updater_entry_count)->default_value(1), "The number of entries to process when profiling updater.")
			("profile_hessian_entry_count", boost::program_options::value<unsigned int>(&profile_hessian_entry_count)->default_value(0), "The number of entries to process when profiling hessian (0 means no limitation).")
			("training_algo", boost::program_options::value<std::string>(&training_algo)->default_value("sdlm"), "Training algorithm (sdlm, sgd).")
			("dump_resume", boost::program_options::value<bool>(&dump_resume)->default_value(true), "Dump neural network data after each epoch.")
			("load_resume,R", boost::program_options::value<bool>(&load_resume)->default_value(false), "Resume neural network training strating from saved.")
			;

		{
			std::vector<string_option> additional_string_options = get_string_options();
			for(std::vector<string_option>::iterator it = additional_string_options.begin(); it != additional_string_options.end(); it++)
			{
				string_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<std::string>(opt.var)->default_value(opt.default_value.c_str()), opt.description.c_str());
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
			std::vector<string_option> additional_string_options = factory->get_string_options();
			for(std::vector<string_option>::iterator it = additional_string_options.begin(); it != additional_string_options.end(); it++)
			{
				string_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<std::string>(opt.var)->default_value(opt.default_value.c_str()), opt.description.c_str());
			}
			std::vector<bool_option> additional_bool_options = factory->get_bool_options();
			for(std::vector<bool_option>::iterator it = additional_bool_options.begin(); it != additional_bool_options.end(); it++)
			{
				bool_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<bool>(opt.var)->default_value(opt.default_value), opt.description.c_str());
			}
			std::vector<float_option> additional_float_options = factory->get_float_options();
			for(std::vector<float_option>::iterator it = additional_float_options.begin(); it != additional_float_options.end(); it++)
			{
				float_option& opt = *it;
				config.add_options()
					(opt.name.c_str(), boost::program_options::value<float>(opt.var)->default_value(opt.default_value), opt.description.c_str());
			}
			std::vector<int_option> additional_int_options = factory->get_int_options();
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

		boost::filesystem::ifstream ifs(config_file);
		if (!ifs)
			throw std::runtime_error((boost::format("Can not open config file %1%") % config_file.string()).str());

		boost::program_options::store(parse_config_file(ifs, config_file_options, true), vm);
		boost::program_options::notify(vm);

		if (vm.count("help"))
		{
			std::cout << visible << "\n";
			return false;
		}

		boost::filesystem::path logfile_path = get_working_data_folder() / logfile_name;
		out_to_log_duplicator_smart_ptr = nnforge_shared_ptr<stream_duplicator>(new stream_duplicator(logfile_path));

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

		factory->initialize();

		tester_factory = factory->create_tester_factory();
		updater_factory = factory->create_updater_factory();
		hessian_factory = factory->create_hessian_factory();
		analyzer_factory = factory->create_analyzer_factory();

		return (action.size() > 0);
	}

	void neural_network_toolset::dump_settings()
	{
		{
			std::cout << "input_data_folder" << "=" << input_data_folder << std::endl;
			std::cout << "working_data_folder" << "=" << working_data_folder << std::endl;
			std::cout << "ann_count" << "=" << ann_count << std::endl;
			std::cout << "training_epoch_count" << "=" << training_epoch_count << std::endl;
			std::cout << "snapshot_count" << "=" << snapshot_count << std::endl;
			std::cout << "snapshot_extension" << "=" << snapshot_extension << std::endl;
			std::cout << "snapshot_mode" << "=" << snapshot_mode << std::endl;
			std::cout << "snapshot_video_fps" << "=" << snapshot_video_fps << std::endl;
			std::cout << "snapshot_ann_index" << "=" << snapshot_ann_index << std::endl;
			std::cout << "mu_increase_factor" << "=" << mu_increase_factor << std::endl;
			std::cout << "max_mu" << "=" << max_mu << std::endl;
			std::cout << "per_layer_mu" << "=" << per_layer_mu << std::endl;
			std::cout << "learning_rate" << "=" << learning_rate << std::endl;
			std::cout << "learning_rate_decay_tail" << "=" << learning_rate_decay_tail_epoch_count << std::endl;
			std::cout << "learning_rate_decay_rate" << "=" << learning_rate_decay_rate << std::endl;
			std::cout << "learning_rate_rise_head" << "=" << learning_rate_rise_head_epoch_count << std::endl;
			std::cout << "learning_rate_rise_rate" << "=" << learning_rate_rise_rate << std::endl;
			std::cout << "batch_offset" << "=" << batch_offset << std::endl;
			std::cout << "test_validate_ann_index" << "=" << test_validate_ann_index << std::endl;
			std::cout << "snapshot_data_set" << "=" << snapshot_data_set << std::endl;
			std::cout << "profile_updater_entry_count" << "=" << profile_updater_entry_count << std::endl;
			std::cout << "profile_hessian_entry_count" << "=" << profile_hessian_entry_count << std::endl;
			std::cout << "training_algo" << "=" << training_algo << std::endl;
			std::cout << "dump_resume" << "=" << dump_resume << std::endl;
			std::cout << "load_resume" << "=" << load_resume << std::endl;
		}
		{
			std::vector<string_option> additional_string_options = get_string_options();
			for(std::vector<string_option>::iterator it = additional_string_options.begin(); it != additional_string_options.end(); it++)
				std::cout << it->name << "=" << *it->var << std::endl;
			std::vector<bool_option> additional_bool_options = get_bool_options();
			for(std::vector<bool_option>::iterator it = additional_bool_options.begin(); it != additional_bool_options.end(); it++)
				std::cout << it->name << "=" << *it->var << std::endl;
			std::vector<float_option> additional_float_options = get_float_options();
			for(std::vector<float_option>::iterator it = additional_float_options.begin(); it != additional_float_options.end(); it++)
				std::cout << it->name << "=" << *it->var << std::endl;
			std::vector<int_option> additional_int_options = get_int_options();
			for(std::vector<int_option>::iterator it = additional_int_options.begin(); it != additional_int_options.end(); it++)
				std::cout << it->name << "=" << *it->var << std::endl;
		}
		{
			std::vector<string_option> additional_string_options = factory->get_string_options();
			for(std::vector<string_option>::iterator it = additional_string_options.begin(); it != additional_string_options.end(); it++)
				std::cout << it->name << "=" << *it->var << std::endl;
			std::vector<bool_option> additional_bool_options = factory->get_bool_options();
			for(std::vector<bool_option>::iterator it = additional_bool_options.begin(); it != additional_bool_options.end(); it++)
				std::cout << it->name << "=" << *it->var << std::endl;
			std::vector<float_option> additional_float_options = factory->get_float_options();
			for(std::vector<float_option>::iterator it = additional_float_options.begin(); it != additional_float_options.end(); it++)
				std::cout << it->name << "=" << *it->var << std::endl;
			std::vector<int_option> additional_int_options = factory->get_int_options();
			for(std::vector<int_option>::iterator it = additional_int_options.begin(); it != additional_int_options.end(); it++)
				std::cout << it->name << "=" << *it->var << std::endl;
		}
	}

	std::string neural_network_toolset::get_action() const
	{
		return action;
	}

	std::vector<string_option> neural_network_toolset::get_string_options()
	{
		return std::vector<string_option>();
	}

	std::vector<bool_option> neural_network_toolset::get_bool_options()
	{
		return std::vector<bool_option>();
	}

	std::vector<float_option> neural_network_toolset::get_float_options()
	{
		return std::vector<float_option>();
	}

	std::vector<int_option> neural_network_toolset::get_int_options()
	{
		return std::vector<int_option>();
	}

	std::string neural_network_toolset::get_class_name_by_id(unsigned int class_id) const
	{
		return (boost::format("%|1$03d|") % class_id).str();
	}

	boost::filesystem::path neural_network_toolset::get_input_data_folder() const
	{
		return input_data_folder;
	}

	boost::filesystem::path neural_network_toolset::get_working_data_folder() const
	{
		return working_data_folder;
	}

	boost::filesystem::path neural_network_toolset::get_ann_subfolder_name() const
	{
		return ann_subfolder_name;
	}

	unsigned int neural_network_toolset::get_epoch_count_for_training_set() const
	{
		return 1;
	}

	network_trainer_smart_ptr neural_network_toolset::get_network_trainer(network_schema_smart_ptr schema) const
	{
		network_trainer_smart_ptr res;

		network_updater_smart_ptr updater = updater_factory->create(
			schema,
			get_error_function(),
			get_dropout_rate_map(),
			get_weight_vector_bound_map());

		if (training_algo == "sdlm")
		{
			hessian_calculator_smart_ptr hessian = hessian_factory->create(schema);

			network_trainer_sdlm_smart_ptr typed_res(
				new network_trainer_sdlm(
					schema,
					hessian,
					updater));

			typed_res->max_mu = max_mu;
			typed_res->per_layer_mu = per_layer_mu;
			typed_res->mu_increase_factor = mu_increase_factor;

			res = typed_res;
		}
		else if (training_algo == "sgd")
		{
			network_trainer_sgd_smart_ptr typed_res(
				new network_trainer_sgd(
					schema,
					updater));

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

		return res;
	}

	std::pair<unsupervised_data_reader_smart_ptr, unsigned int> neural_network_toolset::get_data_reader_and_sample_count_for_snapshots() const
	{
		if (snapshot_data_set == "training")
			return std::make_pair(get_data_reader_for_training(), 1);
		else if (snapshot_data_set == "validating")
		{
			std::pair<supervised_data_reader_smart_ptr, unsigned int> p = get_data_reader_for_validating_and_sample_count();
			return std::make_pair(p.first, p.second);
		}
		else if (snapshot_data_set == "testing")
		{
			if (boost::filesystem::exists(get_working_data_folder() / testing_unsupervised_data_filename))
				return get_data_reader_for_testing_unsupervised_and_sample_count();
			else
			{
				std::pair<supervised_data_reader_smart_ptr, unsigned int> p = get_data_reader_for_testing_supervised_and_sample_count();
				return std::make_pair(p.first, p.second);
			}
		}
		else throw std::runtime_error((boost::format("Unknown data set for taking snapshots: %1%") % snapshot_data_set).str());
	}

	supervised_data_reader_smart_ptr neural_network_toolset::get_original_training_data_reader(const boost::filesystem::path& path) const
	{
		nnforge_shared_ptr<std::istream> in(new boost::filesystem::ifstream(path, std::ios_base::in | std::ios_base::binary));
		supervised_data_reader_smart_ptr reader(new supervised_data_stream_reader(in));
		return reader;
	}

	data_writer_smart_ptr neural_network_toolset::get_randomized_training_data_writer(
		supervised_data_reader& reader,
		const boost::filesystem::path& path) const
	{
		nnforge_shared_ptr<std::ostream> out(new boost::filesystem::ofstream(path, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));
		supervised_data_stream_reader& typed_reader = dynamic_cast<supervised_data_stream_reader&>(reader);
		data_writer_smart_ptr writer(
			new supervised_data_stream_writer(
				out,
				typed_reader.get_input_configuration(),
				typed_reader.get_output_configuration(),
				typed_reader.get_input_type()));
		return writer;
	}

	void neural_network_toolset::randomize_data()
	{
		boost::filesystem::path original_file_path = get_working_data_folder() / training_data_filename;
		supervised_data_reader_smart_ptr reader = get_original_training_data_reader(original_file_path);

		boost::filesystem::path randomized_file_path = get_working_data_folder() / training_randomized_data_filename;
		data_writer_smart_ptr writer = get_randomized_training_data_writer(*reader, randomized_file_path);

		std::cout << "Randomizing " << reader->get_entry_count() << " entries" << std::endl;

		switch(get_network_output_type())
		{
		case network_output_type::type_classifier:
		case network_output_type::type_roc:
			writer->write_randomized_classifier(*reader);
			break;
		default:
			writer->write_randomized(*reader);
			break;
		}
	}

	void neural_network_toolset::create()
	{
		network_schema_smart_ptr schema = get_schema();

		{
			boost::filesystem::ofstream file_with_schema(get_working_data_folder() / schema_filename, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
			schema->write(file_with_schema);
		}
	}

	network_tester_smart_ptr neural_network_toolset::get_tester()
	{
		network_schema_smart_ptr schema(new network_schema());
		{
			boost::filesystem::ifstream in(get_working_data_folder() / schema_filename, std::ios_base::in | std::ios_base::binary);
			schema->read(in);
		}

		return tester_factory->create(schema);
	}

	network_analyzer_smart_ptr neural_network_toolset::get_analyzer()
	{
		network_schema_smart_ptr schema(new network_schema());
		{
			boost::filesystem::ifstream in(get_working_data_folder() / schema_filename, std::ios_base::in | std::ios_base::binary);
			schema->read(in);
		}

		return analyzer_factory->create(schema);
	}

	network_data_smart_ptr neural_network_toolset::load_ann_data(unsigned int ann_id)
	{
		boost::filesystem::path data_filepath = get_working_data_folder() / get_ann_subfolder_name() / (boost::format("ann_trained_%|1$03d|.data") % ann_id).str();
		network_data_smart_ptr data(new network_data());
		{
			boost::filesystem::ifstream in(data_filepath, std::ios_base::in | std::ios_base::binary);
			data->read(in);
		}
		return data;
	}

	std::vector<output_neuron_value_set_smart_ptr> neural_network_toolset::run_batch(
		supervised_data_reader& reader,
		output_neuron_value_set_smart_ptr actual_neuron_value_set)
	{
		network_tester_smart_ptr tester = get_tester();

		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();

		nnforge_regex expression(trained_ann_index_extractor_pattern);
		nnforge_cmatch what;

		std::vector<float> invalid_ratio_list;
		std::vector<output_neuron_value_set_smart_ptr> predicted_neuron_value_set_list;
		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(batch_folder); it != boost::filesystem::directory_iterator(); ++it)
		{
			boost::filesystem::path file_path = it->path();
			std::string file_name = file_path.filename().string();

			if (nnforge_regex_search(file_name.c_str(), what, expression))
			{
				unsigned int index = static_cast<unsigned int>(atol(std::string(what[1].first, what[1].second).c_str()));
				if ((test_validate_ann_index >= 0) && (test_validate_ann_index != index))
					continue;

				network_data_smart_ptr data(new network_data());
				{
					boost::filesystem::ifstream in(file_path, std::ios_base::in | std::ios_base::binary);
					data->read(in);
				}

				tester->set_data(data);

				testing_complete_result_set testing_res(get_error_function(), actual_neuron_value_set);
				tester->test(
					reader,
					testing_res);
				std::cout << "# " << index << ", ";
				get_validating_visualizer()->dump(std::cout, testing_res);
				std::cout << std::endl;

				predicted_neuron_value_set_list.push_back(testing_res.predicted_output_neuron_value_set);
			}
		}

		return predicted_neuron_value_set_list;
	}

	std::vector<output_neuron_value_set_smart_ptr> neural_network_toolset::run_batch(unsupervised_data_reader& reader, unsigned int sample_count)
	{
		network_tester_smart_ptr tester = get_tester();

		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();

		nnforge_regex expression(trained_ann_index_extractor_pattern);
		nnforge_cmatch what;

		std::vector<float> invalid_ratio_list;
		std::vector<output_neuron_value_set_smart_ptr> predicted_neuron_value_set_list;
		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(batch_folder); it != boost::filesystem::directory_iterator(); ++it)
		{
			boost::filesystem::path file_path = it->path();
			std::string file_name = file_path.filename().string();

			if (nnforge_regex_search(file_name.c_str(), what, expression))
			{
				unsigned int index = static_cast<unsigned int>(atol(std::string(what[1].first, what[1].second).c_str()));
				if ((test_validate_ann_index >= 0) && (test_validate_ann_index != index))
					continue;

				network_data_smart_ptr data(new network_data());
				{
					boost::filesystem::ifstream in(file_path, std::ios_base::in | std::ios_base::binary);
					data->read(in);
				}

				tester->set_data(data);

				boost::chrono::steady_clock::time_point start = boost::chrono::high_resolution_clock::now();
				output_neuron_value_set_smart_ptr new_res = tester->run(reader, sample_count);
				boost::chrono::duration<float> sec = boost::chrono::high_resolution_clock::now() - start;
				std::cout << "# " << index;
				std::cout << std::endl;

				predicted_neuron_value_set_list.push_back(new_res);
			}
		}

		return predicted_neuron_value_set_list;
	}

	unsigned int neural_network_toolset::get_testing_sample_count() const
	{
		return 1;
	}

	unsigned int neural_network_toolset::get_validating_sample_count() const
	{
		return 1;
	}

	void neural_network_toolset::validate(bool is_validate)
	{
		if (is_validate || boost::filesystem::exists(get_working_data_folder() / testing_data_filename))
		{
			std::pair<supervised_data_reader_smart_ptr, unsigned int> reader_and_sample_count = is_validate ? get_data_reader_for_validating_and_sample_count() : get_data_reader_for_testing_supervised_and_sample_count();
			output_neuron_value_set_smart_ptr actual_neuron_value_set = reader_and_sample_count.first->get_output_neuron_value_set(reader_and_sample_count.second);
			if (actual_neuron_value_set->neuron_value_list.empty())
				throw neural_network_exception("Empty validating/testing value set");

			std::vector<output_neuron_value_set_smart_ptr> predicted_neuron_value_set_list = run_batch(*reader_and_sample_count.first, actual_neuron_value_set);

			testing_complete_result_set complete_result_set_avg(get_error_function(), actual_neuron_value_set);
			{
				complete_result_set_avg.predicted_output_neuron_value_set = output_neuron_value_set_smart_ptr(new output_neuron_value_set(predicted_neuron_value_set_list, output_neuron_value_set::merge_average));
				complete_result_set_avg.recalculate_mse();
				std::cout << "Merged (average), ";
				get_validating_visualizer()->dump(std::cout, complete_result_set_avg);
				std::cout << std::endl;
			}
		}
		else if (boost::filesystem::exists(get_working_data_folder() / testing_unsupervised_data_filename))
		{
			std::pair<unsupervised_data_reader_smart_ptr, unsigned int> reader_and_sample_count = get_data_reader_for_testing_unsupervised_and_sample_count();

			std::vector<output_neuron_value_set_smart_ptr> predicted_neuron_value_set_list = run_batch(*reader_and_sample_count.first, reader_and_sample_count.second);

			run_test_with_unsupervised_data(predicted_neuron_value_set_list);
		}
		else throw neural_network_exception((boost::format("File %1% doesn't exist - nothing to test") % (get_working_data_folder() / testing_unsupervised_data_filename).string()).str());
	}

	void neural_network_toolset::run_test_with_unsupervised_data(std::vector<output_neuron_value_set_smart_ptr>& predicted_neuron_value_set_list)
	{
		throw neural_network_exception("Running test with unsupervised data is not implemented by the derived toolset");
	}

	void neural_network_toolset::generate_input_normalizer()
	{
		nnforge_shared_ptr<std::istream> in(new boost::filesystem::ifstream(get_working_data_folder() / training_data_filename, std::ios_base::in | std::ios_base::binary));
		nnforge::supervised_data_stream_reader reader(in);

		std::vector<nnforge::feature_map_data_stat> feature_map_data_stat_list = reader.get_feature_map_input_data_stat_list();
		unsigned int feature_map_id = 0;
		for(std::vector<nnforge::feature_map_data_stat>::const_iterator it = feature_map_data_stat_list.begin(); it != feature_map_data_stat_list.end(); ++it, ++feature_map_id)
			std::cout << "Feature map # " << feature_map_id << ": " << *it << std::endl;

		normalize_data_transformer normalizer(feature_map_data_stat_list);

		boost::filesystem::ofstream file_with_schema(get_working_data_folder() / normalizer_input_filename, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
		normalizer.write(file_with_schema);
	}

	void neural_network_toolset::generate_output_normalizer()
	{
		nnforge_shared_ptr<std::istream> in(new boost::filesystem::ifstream(get_working_data_folder() / training_data_filename, std::ios_base::in | std::ios_base::binary));
		nnforge::supervised_data_stream_reader reader(in);

		std::vector<nnforge::feature_map_data_stat> feature_map_data_stat_list = reader.get_feature_map_output_data_stat_list();
		unsigned int feature_map_id = 0;
		for(std::vector<nnforge::feature_map_data_stat>::const_iterator it = feature_map_data_stat_list.begin(); it != feature_map_data_stat_list.end(); ++it, ++feature_map_id)
			std::cout << "Feature map # " << feature_map_id << ": " << *it << std::endl;

		normalize_data_transformer normalizer(feature_map_data_stat_list);

		boost::filesystem::ofstream file_with_schema(get_working_data_folder() / normalizer_output_filename, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
		normalizer.write(file_with_schema);
	}

	normalize_data_transformer_smart_ptr neural_network_toolset::get_input_data_normalize_transformer() const
	{
		boost::filesystem::path normalizer_filepath = get_working_data_folder() / normalizer_input_filename;
		if (!boost::filesystem::exists(normalizer_filepath))
			throw neural_network_exception((boost::format("Normalizer file not found: %1%") % normalizer_filepath.string()).str());

		normalize_data_transformer_smart_ptr res(new normalize_data_transformer());

		boost::filesystem::ifstream file_with_schema(normalizer_filepath, std::ios_base::in | std::ios_base::binary);
		res->read(file_with_schema);

		return res;
	}

	normalize_data_transformer_smart_ptr neural_network_toolset::get_output_data_normalize_transformer() const
	{
		boost::filesystem::path normalizer_filepath = get_working_data_folder() / normalizer_output_filename;
		if (!boost::filesystem::exists(normalizer_filepath))
			throw neural_network_exception((boost::format("Normalizer file not found: %1%") % normalizer_filepath.string()).str());

		normalize_data_transformer_smart_ptr res(new normalize_data_transformer());

		boost::filesystem::ifstream file_with_schema(normalizer_filepath, std::ios_base::in | std::ios_base::binary);
		res->read(file_with_schema);

		return res;
	}

	normalize_data_transformer_smart_ptr neural_network_toolset::get_reverse_output_data_normalize_transformer() const
	{
		return get_output_data_normalize_transformer()->get_inverted_transformer();
	}

	std::vector<std::vector<std::pair<unsigned int, unsigned int> > > neural_network_toolset::get_samples_for_snapshot(
		network_data_smart_ptr data,
		unsupervised_data_reader_smart_ptr reader,
		unsigned int sample_count)
	{
		std::vector<std::vector<std::pair<unsigned int, unsigned int> > > layer_feature_map_sample_and_offset_list;
		std::vector<std::vector<float> > layer_feature_map_max_val_list;

		reader->reset();

		network_tester_smart_ptr tester = get_tester();
		tester->set_data(data);
		tester->set_input_configuration_specific(reader->get_input_configuration());

		std::vector<unsigned char> input(reader->get_input_configuration().get_neuron_count() * reader->get_input_neuron_elem_size());
		unsigned int current_sample_id = 0;
		while(true)
		{
			if (!reader->read(&(*input.begin())))
				break;
			for(unsigned int skip_id = 0; skip_id < sample_count - 1; ++skip_id)
				if (!reader->read(0))
					break;

			std::vector<layer_configuration_specific_snapshot_smart_ptr> current_snapshot = tester->get_snapshot(
				&(*input.begin()),
				reader->get_input_type(),
				reader->get_input_configuration().get_neuron_count());

			if (current_sample_id == 0)
			{
				for(std::vector<layer_configuration_specific_snapshot_smart_ptr>::const_iterator it = current_snapshot.begin() + 1; it != current_snapshot.end(); ++it)
				{
					const layer_configuration_specific_snapshot& sn = **it;
					unsigned int neuron_count_per_feature_map = sn.config.get_neuron_count_per_feature_map();

					std::vector<std::pair<unsigned int, unsigned int> > feature_map_sample_and_offset_list;
					std::vector<float> feature_map_max_val_list;
					for(std::vector<float>::const_iterator src_it = sn.data.begin(); src_it != sn.data.end(); src_it += neuron_count_per_feature_map)
					{
						std::vector<float>::const_iterator max_it = std::max_element(src_it, src_it + neuron_count_per_feature_map);
						unsigned int offset = max_it - src_it;
						float val = *max_it;

						feature_map_sample_and_offset_list.push_back(std::make_pair(0, offset));
						feature_map_max_val_list.push_back(val);
					}

					layer_feature_map_sample_and_offset_list.push_back(feature_map_sample_and_offset_list);
					layer_feature_map_max_val_list.push_back(feature_map_max_val_list);
				}
			}
			else
			{
				std::vector<std::vector<std::pair<unsigned int, unsigned int> > >::iterator layer_feature_map_sample_and_offset_it = layer_feature_map_sample_and_offset_list.begin();
				std::vector<std::vector<float> >::iterator layer_feature_map_max_val_it = layer_feature_map_max_val_list.begin();
				for(std::vector<layer_configuration_specific_snapshot_smart_ptr>::const_iterator it = current_snapshot.begin() + 1; it != current_snapshot.end(); ++it, ++layer_feature_map_sample_and_offset_it, ++layer_feature_map_max_val_it)
				{
					const layer_configuration_specific_snapshot& sn = **it;
					std::vector<std::pair<unsigned int, unsigned int> >& current_feature_map_sample_and_offset = *layer_feature_map_sample_and_offset_it;
					std::vector<float>& current_feature_map_max_val = *layer_feature_map_max_val_it;
					unsigned int neuron_count_per_feature_map = sn.config.get_neuron_count_per_feature_map();

					std::vector<std::pair<unsigned int, unsigned int> >::iterator sample_and_offset_it = current_feature_map_sample_and_offset.begin();
					std::vector<float>::iterator max_val_it = current_feature_map_max_val.begin();
					for(std::vector<float>::const_iterator src_it = sn.data.begin(); src_it != sn.data.end(); src_it += neuron_count_per_feature_map, ++sample_and_offset_it, ++max_val_it)
					{
						std::vector<float>::const_iterator max_it = std::max_element(src_it, src_it + neuron_count_per_feature_map);
						unsigned int offset = max_it - src_it;
						float val = *max_it;

						if (val < *max_val_it)
						{
							*max_val_it = val;
							sample_and_offset_it->first = current_sample_id;
							sample_and_offset_it->second = offset;
						}
					}
				}
			}

			++current_sample_id;
		}

		return layer_feature_map_sample_and_offset_list;
	}

	struct sample_location
	{
		unsigned int layer_id;
		unsigned int feature_map_id;
		unsigned int offset;
	};

	void neural_network_toolset::snapshot_data()
	{
		boost::filesystem::path snapshot_folder = get_working_data_folder() / snapshot_data_subfolder_name;
		boost::filesystem::create_directories(snapshot_folder);

		std::pair<unsupervised_data_reader_smart_ptr, unsigned int> reader_and_sample_count = get_data_reader_and_sample_count_for_snapshots();

		unsupervised_data_reader_smart_ptr reader = reader_and_sample_count.first;
		unsigned int reader_sample_count = reader_and_sample_count.second;

		std::cout << "Parsing entries from " << snapshot_data_set << " data set..." << std::endl;

		layer_configuration_specific_snapshot sn(reader->get_input_configuration());
		for(unsigned int i = 0; i < snapshot_count; ++i)
		{
			if (reader->get_input_type() == neuron_data_type::type_float)
			{
				if (!reader->read(&(*sn.data.begin())))
					break;
			}
			else if (reader->get_input_type() == neuron_data_type::type_byte)
			{
				std::vector<unsigned char> buf(sn.data.size());
				if (!reader->read(&(*buf.begin())))
					break;
				for(int i = 0; i < sn.data.size(); ++i)
					sn.data[i] = static_cast<float>(buf[i]) * (1.0F / 255.0F);
			}

			boost::filesystem::path snapshot_file_path = snapshot_folder / (boost::format("snapshot_%|1$05d|.%2%") % i % snapshot_extension).str();

			snapshot_visualizer::save_2d_snapshot(
				sn,
				snapshot_file_path.string().c_str(),
				is_rgb_input() && (sn.config.feature_map_count == 3),
				true);
		}
	}

	void neural_network_toolset::snapshot()
	{
		boost::filesystem::path snapshot_folder = get_working_data_folder() / snapshot_subfolder_name;
		boost::filesystem::create_directories(snapshot_folder);

		network_data_smart_ptr data = load_ann_data(snapshot_ann_index);

		std::pair<unsupervised_data_reader_smart_ptr, unsigned int> reader_and_sample_count = get_data_reader_and_sample_count_for_snapshots();
		unsupervised_data_reader_smart_ptr reader = reader_and_sample_count.first;
		unsigned int reader_sample_count = reader_and_sample_count.second;

		std::cout << "Parsing entries from " << snapshot_data_set << " data set..." << std::endl;
		std::vector<std::vector<std::pair<unsigned int, unsigned int> > > layer_feature_map_sample_and_offset_list = get_samples_for_snapshot(
			data,
			reader,
			reader_sample_count);

		std::map<unsigned int, std::vector<sample_location> > sample_to_location_list_map;
		unsigned int snapshot_count = 0;
		for(unsigned int layer_id = 0; layer_id < layer_feature_map_sample_and_offset_list.size(); ++layer_id)
		{
			const std::vector<std::pair<unsigned int, unsigned int> >& feature_map_sample_and_offset_list = layer_feature_map_sample_and_offset_list[layer_id];
			for(unsigned int feature_map_id = 0; feature_map_id < feature_map_sample_and_offset_list.size(); ++feature_map_id)
			{
				unsigned int sample_id = feature_map_sample_and_offset_list[feature_map_id].first;
				unsigned int offset = feature_map_sample_and_offset_list[feature_map_id].second;

				sample_location new_item;
				new_item.layer_id = layer_id;
				new_item.feature_map_id = feature_map_id;
				new_item.offset = offset;
				std::map<unsigned int, std::vector<sample_location> >::iterator it = sample_to_location_list_map.find(sample_id);
				if (it == sample_to_location_list_map.end())
					sample_to_location_list_map.insert(std::make_pair(sample_id, std::vector<sample_location>(1, new_item)));
				else
					it->second.push_back(new_item);
				++snapshot_count;
			}
		}
		std::cout << snapshot_count << " snapshots from " << sample_to_location_list_map.size() << " samples" << std::endl;

		reader->reset();

		network_schema_smart_ptr schema(new network_schema());
		{
			boost::filesystem::ifstream in(get_working_data_folder() / schema_filename, std::ios_base::in | std::ios_base::binary);
			schema->read(in);
		}
		layer_configuration_specific_list layer_config_list = schema->get_layer_configuration_specific_list(reader->get_input_configuration());

		network_analyzer_smart_ptr analyzer = get_analyzer();
		analyzer->set_data(data);
		analyzer->set_input_configuration_specific(reader->get_input_configuration());

		std::vector<unsigned char> input(reader->get_input_configuration().get_neuron_count() * reader->get_input_neuron_elem_size());

		unsigned int current_sample_id = 0;
		while(true)
		{
			std::map<unsigned int, std::vector<sample_location> >::iterator it = sample_to_location_list_map.find(current_sample_id);
			if (it == sample_to_location_list_map.end())
			{
				if (!reader->read(0))
					break;
			}
			else
			{
				if (!reader->read(&(*input.begin())))
					break;

				analyzer->set_input_data(
					&(*input.begin()),
					reader->get_input_type(),
					reader->get_input_configuration().get_neuron_count());

				const std::vector<sample_location>& location_list = it->second;
				for(std::vector<sample_location>::const_iterator sample_location_it = location_list.begin(); sample_location_it != location_list.end(); ++sample_location_it)
				{
					unsigned int layer_id = sample_location_it->layer_id;
					unsigned int feature_map_id = sample_location_it->feature_map_id;
					unsigned int offset = sample_location_it->offset;

					boost::filesystem::path snapshot_layer_folder = snapshot_folder / (boost::format("%|1$03d|") % layer_id).str();
					boost::filesystem::create_directories(snapshot_layer_folder);

					boost::filesystem::path snapshot_file_path = snapshot_layer_folder / ((boost::format("snapshot_layer_%|1$03d|_fm_%|2$04d|_%3%_sample_%4%") % layer_id % feature_map_id % snapshot_data_set % current_sample_id).str() + "." + snapshot_extension);
					boost::filesystem::path original_snapshot_file_path = snapshot_layer_folder / ((boost::format("snapshot_layer_%|1$03d|_fm_%|2$04d|_%3%_sample_%4%_original") % layer_id % feature_map_id % snapshot_data_set % current_sample_id).str() + "." + snapshot_extension);

					std::pair<layer_configuration_specific_snapshot_smart_ptr, layer_configuration_specific_snapshot_smart_ptr> input_image_pair = run_analyzer_for_single_neuron(
						*analyzer,
						layer_id,
						feature_map_id,
						layer_config_list[layer_id + 1].get_offsets(offset),
						layer_config_list[layer_id + 1].feature_map_count);

					snapshot_visualizer::save_2d_snapshot(
						*(input_image_pair.first),
						snapshot_file_path.string().c_str(),
						is_rgb_input() && (input_image_pair.first->config.feature_map_count == 3),
						true);

					snapshot_visualizer::save_2d_snapshot(
						*(input_image_pair.second),
						original_snapshot_file_path.string().c_str(),
						is_rgb_input() && (input_image_pair.second->config.feature_map_count == 3),
						false);
				}
			}

			for(unsigned int skip_id = 0; skip_id < reader_sample_count - 1; ++skip_id)
				if (!reader->read(0))
					break;
			++current_sample_id;
		}
	}

	std::pair<layer_configuration_specific_snapshot_smart_ptr, layer_configuration_specific_snapshot_smart_ptr> neural_network_toolset::run_analyzer_for_single_neuron(
		network_analyzer& analyzer,
		unsigned int layer_id,
		unsigned int feature_map_id,
		const std::vector<unsigned int>& location_list,
		unsigned int feature_map_count) const
	{
		layer_configuration_specific_snapshot output_data;
		output_data.config = layer_configuration_specific(feature_map_count, std::vector<unsigned int>(location_list.size(), 1));
		output_data.data.resize(feature_map_count, 0.0F);
		output_data.data[feature_map_id] = 1.0F;

		return analyzer.run_backprop(
			output_data,
			location_list,
			layer_id);
	}

	void neural_network_toolset::ann_snapshot()
	{
		network_schema_smart_ptr schema(new network_schema());
		{
			boost::filesystem::ifstream in(get_working_data_folder() / schema_filename, std::ios_base::in | std::ios_base::binary);
			schema->read(in);
		}
		std::vector<layer_data_configuration_list> layer_data_configuration_list_list = schema->get_layer_data_configuration_list_list();

		network_data_smart_ptr data = load_ann_data(snapshot_ann_index);

		std::string ann_snapshot_filename = "trained";
		save_ann_snapshot(ann_snapshot_filename, *data, layer_data_configuration_list_list);
	}

	void neural_network_toolset::save_ann_snapshot(
		const std::string& name,
		const network_data& data,
		const std::vector<layer_data_configuration_list>& layer_data_configuration_list_list)
	{
		boost::filesystem::path ann_snapshot_folder = get_working_data_folder() / ann_snapshot_subfolder_name;
		boost::filesystem::create_directories(ann_snapshot_folder);

		boost::filesystem::path ann_snapshot_file_path = ann_snapshot_folder / (std::string("ann_snapshot_") + name + "." + snapshot_extension);

		snapshot_visualizer::save_ann_snapshot(data, layer_data_configuration_list_list, ann_snapshot_file_path.string().c_str());
	}

	void neural_network_toolset::snapshot_invalid()
	{
		network_tester_smart_ptr tester = get_tester();

		network_data_smart_ptr data = load_ann_data(snapshot_ann_index);

		tester->set_data(data);

		std::pair<supervised_data_reader_smart_ptr, unsigned int> reader_and_sample_count = get_data_reader_for_validating_and_sample_count();
		output_neuron_value_set_smart_ptr actual_neuron_value_set = reader_and_sample_count.first->get_output_neuron_value_set(reader_and_sample_count.second);

		testing_complete_result_set testing_res(get_error_function(), actual_neuron_value_set);
		if (reader_and_sample_count.first->get_output_configuration().get_neuron_count() == 1)
			throw "Invalid snapshots is not implemented for single output neuron configuration";
		tester->test(
			*reader_and_sample_count.first,
			testing_res);

		output_neuron_class_set predicted_cs(*testing_res.predicted_output_neuron_value_set, 1);
		output_neuron_class_set actual_cs(*testing_res.actual_output_neuron_value_set, 1);
		classifier_result cr(predicted_cs, actual_cs);

		reader_and_sample_count.first->reset();

		tester->set_input_configuration_specific(reader_and_sample_count.first->get_input_configuration());

		std::vector<unsigned char> input(reader_and_sample_count.first->get_input_configuration().get_neuron_count() * reader_and_sample_count.first->get_input_neuron_elem_size());
		unsigned int entry_id = 0;
		std::vector<unsigned int>::const_iterator actual_it = cr.actual_class_id_list.begin();
		for(std::vector<unsigned int>::const_iterator it = cr.predicted_class_id_list.begin();
			it != cr.predicted_class_id_list.end();
			++it, ++actual_it)
		{
			if (!reader_and_sample_count.first->read(&(*input.begin()), 0))
				throw std::runtime_error("Not enough entries");
			for(unsigned int i = 1; i < reader_and_sample_count.second; ++i)
				if (!reader_and_sample_count.first->read(0, 0))
					throw std::runtime_error("Not enough entries");

			unsigned int predicted_class_id = *it;
			unsigned int actual_class_id = *actual_it;

			if (predicted_class_id != actual_class_id)
			{
				std::cout << "Actual: " << get_class_name_by_id(actual_class_id) << ", Predicted: " << get_class_name_by_id(predicted_class_id) << ", Entry ID: " << entry_id << std::endl;
			}

			entry_id++;
		}
	}

	unsigned int neural_network_toolset::get_classifier_visualizer_top_n() const
	{
		return 1;
	}

	unsigned int neural_network_toolset::get_starting_index_for_batch_training()
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

	std::vector<network_data_pusher_smart_ptr> neural_network_toolset::get_validators_for_training(network_schema_smart_ptr schema)
	{
		std::vector<network_data_pusher_smart_ptr> res;

		if (is_training_with_validation())
		{
			std::pair<supervised_data_reader_smart_ptr, unsigned int> validating_data_reader_and_sample_count = get_data_reader_for_validating_and_sample_count();
			res.push_back(network_data_pusher_smart_ptr(new validate_progress_network_data_pusher(
				tester_factory->create(schema),
				validating_data_reader_and_sample_count.first,
				get_validating_visualizer(),
				get_error_function(),
				validating_data_reader_and_sample_count.second)));
		}

		return res;
	}

	supervised_data_reader_smart_ptr neural_network_toolset::get_data_reader_for_training() const
	{
		supervised_data_reader_smart_ptr current_reader = get_initial_data_reader_for_training();

		unsigned int epoch_count = get_epoch_count_for_training_set();
		if (epoch_count > 1)
		{
			supervised_data_reader_smart_ptr new_reader(new supervised_multiple_epoch_data_reader(current_reader, epoch_count));
			current_reader = new_reader;
		}

		{
			std::vector<data_transformer_smart_ptr> data_transformer_list = get_input_data_transformer_list_for_training();
			for(std::vector<data_transformer_smart_ptr>::iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
			{
				supervised_data_reader_smart_ptr new_reader(new supervised_transformed_input_data_reader(current_reader, *it));
				current_reader = new_reader;
			}
		}
		{
			std::vector<data_transformer_smart_ptr> data_transformer_list = get_output_data_transformer_list_for_training();
			for(std::vector<data_transformer_smart_ptr>::iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
			{
				supervised_data_reader_smart_ptr new_reader(new supervised_transformed_output_data_reader(current_reader, *it));
				current_reader = new_reader;
			}
		}
		return current_reader;
	}

	supervised_data_reader_smart_ptr neural_network_toolset::get_initial_data_reader_for_training() const
	{
		nnforge_shared_ptr<std::istream> training_data_stream(new boost::filesystem::ifstream(get_working_data_folder() / training_randomized_data_filename, std::ios_base::in | std::ios_base::binary));
		supervised_data_reader_smart_ptr current_reader(new supervised_data_stream_reader(training_data_stream));
		return current_reader;
	}

	std::pair<supervised_data_reader_smart_ptr, unsigned int> neural_network_toolset::get_data_reader_for_validating_and_sample_count() const
	{
		supervised_data_reader_smart_ptr current_reader = get_initial_data_reader_for_validating();

		unsigned int sample_count = get_validating_sample_count();
		{
			std::vector<data_transformer_smart_ptr> data_transformer_list = get_input_data_transformer_list_for_validating();
			for(std::vector<data_transformer_smart_ptr>::iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
			{
				supervised_data_reader_smart_ptr new_reader(new supervised_transformed_input_data_reader(current_reader, *it));
				sample_count *= (*it)->get_sample_count();
				current_reader = new_reader;
			}
		}
		{
			std::vector<data_transformer_smart_ptr> data_transformer_list = get_output_data_transformer_list_for_validating();
			for(std::vector<data_transformer_smart_ptr>::iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
			{
				supervised_data_reader_smart_ptr new_reader(new supervised_transformed_output_data_reader(current_reader, *it));
				sample_count *= (*it)->get_sample_count();
				current_reader = new_reader;
			}
		}

		return std::make_pair(current_reader, sample_count);
	}

	supervised_data_reader_smart_ptr neural_network_toolset::get_initial_data_reader_for_validating() const
	{
		nnforge_shared_ptr<std::istream> validating_data_stream(new boost::filesystem::ifstream(get_working_data_folder() / validating_data_filename, std::ios_base::in | std::ios_base::binary));
		supervised_data_reader_smart_ptr current_reader(new supervised_data_stream_reader(validating_data_stream));
		return current_reader;
	}

	std::pair<supervised_data_reader_smart_ptr, unsigned int> neural_network_toolset::get_data_reader_for_testing_supervised_and_sample_count() const
	{
		supervised_data_reader_smart_ptr current_reader = get_initial_data_reader_for_testing_supervised();
		unsigned int sample_count = get_testing_sample_count();
		{
			std::vector<data_transformer_smart_ptr> data_transformer_list = get_input_data_transformer_list_for_testing();
			for(std::vector<data_transformer_smart_ptr>::iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
			{
				supervised_data_reader_smart_ptr new_reader(new supervised_transformed_input_data_reader(current_reader, *it));
				sample_count *= (*it)->get_sample_count();
				current_reader = new_reader;
			}
		}
		{
			std::vector<data_transformer_smart_ptr> data_transformer_list = get_output_data_transformer_list_for_testing();
			for(std::vector<data_transformer_smart_ptr>::iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
			{
				supervised_data_reader_smart_ptr new_reader(new supervised_transformed_output_data_reader(current_reader, *it));
				sample_count *= (*it)->get_sample_count();
				current_reader = new_reader;
			}
		}

		return std::make_pair(current_reader, sample_count);
	}

	supervised_data_reader_smart_ptr neural_network_toolset::get_initial_data_reader_for_testing_supervised() const
	{
		nnforge_shared_ptr<std::istream> testing_data_stream(new boost::filesystem::ifstream(get_working_data_folder() / testing_data_filename, std::ios_base::in | std::ios_base::binary));
		supervised_data_reader_smart_ptr current_reader(new supervised_data_stream_reader(testing_data_stream));
		return current_reader;
	}

	std::pair<unsupervised_data_reader_smart_ptr, unsigned int> neural_network_toolset::get_data_reader_for_testing_unsupervised_and_sample_count() const
	{
		unsupervised_data_reader_smart_ptr current_reader = get_initial_data_reader_for_testing_unsupervised();
		unsigned int sample_count = get_testing_sample_count();
		{
			std::vector<data_transformer_smart_ptr> data_transformer_list = get_input_data_transformer_list_for_testing();
			for(std::vector<data_transformer_smart_ptr>::iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
			{
				unsupervised_data_reader_smart_ptr new_reader(new unsupervised_transformed_input_data_reader(current_reader, *it));
				sample_count *= (*it)->get_sample_count();
				current_reader = new_reader;
			}
		}

		return std::make_pair(current_reader, sample_count);
	}

	unsupervised_data_reader_smart_ptr neural_network_toolset::get_initial_data_reader_for_testing_unsupervised() const
	{
		nnforge_shared_ptr<std::istream> testing_data_stream(new boost::filesystem::ifstream(get_working_data_folder() / testing_unsupervised_data_filename, std::ios_base::in | std::ios_base::binary));
		unsupervised_data_reader_smart_ptr current_reader(new unsupervised_data_stream_reader(testing_data_stream));
		return current_reader;
	}

	void neural_network_toolset::train()
	{
		network_schema_smart_ptr schema(new network_schema());
		{
			boost::filesystem::ifstream in(get_working_data_folder() / schema_filename, std::ios_base::in | std::ios_base::binary);
			schema->read(in);
		}

		network_trainer_smart_ptr trainer = get_network_trainer(schema);

		supervised_data_reader_smart_ptr training_data_reader = get_data_reader_for_training();

		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();
		boost::filesystem::create_directories(batch_folder);
		boost::filesystem::path batch_resume_folder = batch_folder / ann_resume_subfolder_name;
		boost::filesystem::create_directories(batch_resume_folder);

		nnforge_shared_ptr<network_data_peeker> peeker;
		if (load_resume)
		{
			peeker = nnforge_shared_ptr<network_data_peeker>(new network_data_peeker_load_resume(batch_folder, batch_resume_folder));
		}
		else
		{
			unsigned int starting_index = get_starting_index_for_batch_training();
			peeker = nnforge_shared_ptr<network_data_peeker>(new network_data_peeker_random(ann_count, starting_index));
		}

		complex_network_data_pusher progress;

		if (dump_resume)
		{
			progress.push_back(network_data_pusher_smart_ptr(new save_resume_network_data_pusher(batch_resume_folder)));
		}

		progress.push_back(network_data_pusher_smart_ptr(new report_progress_network_data_pusher()));

		std::vector<network_data_pusher_smart_ptr> validators_for_training = get_validators_for_training(schema);
		progress.insert(progress.end(), validators_for_training.begin(), validators_for_training.end());

		summarize_network_data_pusher res(batch_folder);

		trainer->train(
			*training_data_reader,
			*peeker,
			progress,
			res);
	}

	void neural_network_toolset::profile_updater()
	{
		network_schema_smart_ptr schema(new network_schema());
		{
			boost::filesystem::ifstream in(get_working_data_folder() / schema_filename, std::ios_base::in | std::ios_base::binary);
			schema->read(in);
		}

		network_updater_smart_ptr updater = updater_factory->create(
			schema,
			get_error_function(),
			get_dropout_rate_map(),
			get_weight_vector_bound_map());

		supervised_data_reader_smart_ptr training_data_reader = get_data_reader_for_training();
		training_data_reader = supervised_data_reader_smart_ptr(new supervised_limited_entry_count_data_reader(training_data_reader, profile_updater_entry_count));

		std::vector<network_data_smart_ptr> learning_rates(ann_count);
		std::vector<network_data_smart_ptr> data(ann_count);

		{
			random_generator data_gen = rnd::get_random_generator(47597);
			for(int i = ann_count - 1; i >= 0; --i)
			{
				network_data_smart_ptr data_elem(new network_data(*schema));
				data_elem->randomize(*schema, data_gen);
				data[i] = data_elem;
			}
		}

		{
			random_generator data_gen = rnd::get_random_generator(674578);
			for(int i = ann_count - 1; i >= 0; --i)
			{
				network_data_smart_ptr ts(new network_data(*schema));
				ts->random_fill(learning_rate * 0.5F, learning_rate * 1.5F, data_gen);
				//ts->fill(learning_rate);
				learning_rates[i] = ts;
			}
		}

		std::vector<float> random_uniform_list(1 << 10);
		random_generator gen = rnd::get_random_generator();
		nnforge_uniform_real_distribution<float> dist(0.0F, 1.0F);
		for(std::vector<float>::iterator it = random_uniform_list.begin(); it != random_uniform_list.end(); ++it)
			*it = dist(gen);

		boost::chrono::steady_clock::time_point start = boost::chrono::high_resolution_clock::now();
		updater->update(
			*training_data_reader,
			learning_rates,
			data);
		boost::chrono::duration<float> sec = boost::chrono::high_resolution_clock::now() - start;
		/*
		{
			boost::filesystem::create_directories("profile_data");
			for(network_data::const_iterator it = data[0]->begin(); it != data[0]->end(); it++)
			{
				for(layer_data::const_iterator it2 = (*it)->begin(); it2 != (*it)->end(); it2++)
				{
					if (!it2->empty())
					{
						std::string filename = (boost::format("%1%_%|2$02d|_%|3$02d|.txt") % "data" % (it - data[0]->begin()) % (it2 - (*it)->begin())).str();
						std::ofstream out((boost::filesystem::path("profile_data") / filename).string()); 
						for(std::vector<float>::const_iterator it3 = it2->begin(); it3 != it2->end(); ++it3)
							out << *it3 << std::endl;
					}
				}
			}
		}
		*/
		float time_to_complete_seconds = sec.count();

		if (time_to_complete_seconds != 0.0F)
		{
			float flops = static_cast<float>(training_data_reader->get_entry_count()) * updater->get_flops_for_single_entry() * static_cast<float>(ann_count);
			float gflops = flops / time_to_complete_seconds * 1.0e-9F;
			std::cout << (boost::format("%|1$.1f| GFLOPs, %|2$.2f| seconds") % gflops % time_to_complete_seconds) << std::endl;
		}

		std::cout << data[data.size()-1]->get_stat() << std::endl;
	}

	void neural_network_toolset::profile_hessian()
	{
		network_schema_smart_ptr schema(new network_schema());
		{
			boost::filesystem::ifstream in(get_working_data_folder() / schema_filename, std::ios_base::in | std::ios_base::binary);
			schema->read(in);
		}

		hessian_calculator_smart_ptr hessian = hessian_factory->create(schema);

		supervised_data_reader_smart_ptr training_data_reader = get_data_reader_for_training();

		network_data_smart_ptr data(new network_data(*schema));
		{
			random_generator gen = rnd::get_random_generator(47597);
			data->randomize(
				*schema,
				gen);
		}

		unsigned int hessian_entry_count = std::max(static_cast<unsigned int>(0.05F * training_data_reader->get_entry_count()), 50U);
		if (profile_hessian_entry_count > 0)
			hessian_entry_count = profile_hessian_entry_count;
		unsigned int hessian_entry_to_process_count = std::min<unsigned int>(hessian_entry_count, training_data_reader->get_entry_count());
		boost::chrono::steady_clock::time_point start = boost::chrono::high_resolution_clock::now();
		network_data_smart_ptr hessian_data = hessian->get_hessian(
			*training_data_reader,
			data,
			hessian_entry_to_process_count);
		boost::chrono::duration<float> sec = boost::chrono::high_resolution_clock::now() - start;
		float time_to_complete_seconds = sec.count();

		if (time_to_complete_seconds != 0.0F)
		{
			float flops = static_cast<float>(hessian_entry_to_process_count) * hessian->get_flops_for_single_entry();
			float gflops = flops / time_to_complete_seconds * 1.0e-9F;
			std::cout << (boost::format("%|1$.1f| GFLOPs, %|2$.2f| seconds") % gflops % time_to_complete_seconds) << std::endl;
		}

		std::cout << hessian_data->get_stat() << std::endl;
	}

	network_output_type::output_type neural_network_toolset::get_network_output_type() const
	{
		return network_output_type::type_classifier;
	}

	testing_complete_result_set_visualizer_smart_ptr neural_network_toolset::get_validating_visualizer() const
	{
		switch (get_network_output_type())
		{
		case network_output_type::type_classifier:
			return testing_complete_result_set_visualizer_smart_ptr(new testing_complete_result_set_classifier_visualizer(get_classifier_visualizer_top_n()));
		case network_output_type::type_roc:
			return testing_complete_result_set_visualizer_smart_ptr(new testing_complete_result_set_roc_visualizer());
		default:
			return testing_complete_result_set_visualizer_smart_ptr(new testing_complete_result_set_visualizer());
		}
	}

	bool neural_network_toolset::is_training_with_validation() const
	{
		return true;
	}

	std::map<unsigned int, float> neural_network_toolset::get_dropout_rate_map() const
	{
		return std::map<unsigned int, float>();
	}

	std::map<unsigned int, weight_vector_bound> neural_network_toolset::get_weight_vector_bound_map() const
	{
		return std::map<unsigned int, weight_vector_bound>();
	}

	std::vector<data_transformer_smart_ptr> neural_network_toolset::get_input_data_transformer_list_for_training() const
	{
		return std::vector<data_transformer_smart_ptr>();
	}

	std::vector<data_transformer_smart_ptr> neural_network_toolset::get_output_data_transformer_list_for_training() const
	{
		return std::vector<data_transformer_smart_ptr>();
	}

	std::vector<data_transformer_smart_ptr> neural_network_toolset::get_input_data_transformer_list_for_validating() const
	{
		return std::vector<data_transformer_smart_ptr>();
	}

	std::vector<data_transformer_smart_ptr> neural_network_toolset::get_output_data_transformer_list_for_validating() const
	{
		return std::vector<data_transformer_smart_ptr>();
	}

	std::vector<data_transformer_smart_ptr> neural_network_toolset::get_input_data_transformer_list_for_testing() const
	{
		return std::vector<data_transformer_smart_ptr>();
	}

	std::vector<data_transformer_smart_ptr> neural_network_toolset::get_output_data_transformer_list_for_testing() const
	{
		return std::vector<data_transformer_smart_ptr>();
	}

	const_error_function_smart_ptr neural_network_toolset::get_error_function() const
	{
		return error_function_smart_ptr(new mse_error_function());
	}

	bool neural_network_toolset::is_rgb_input() const
	{
		return true;
	}
}
