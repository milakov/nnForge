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

#include "neural_network_toolset.h"

#include <boost/program_options.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/format.hpp>
#include <boost/chrono.hpp>
#include <boost/algorithm/string.hpp>

#include <regex>
#include <algorithm>
#include <numeric>
#include <limits>

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
#include "debug_util.h"
#include "supervised_shuffle_entries_data_reader.h"

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
	const char * neural_network_toolset::output_subfolder_name = "output";
	const char * neural_network_toolset::output_neurons_filename = "output_neurons.dt";
	const char * neural_network_toolset::mixture_filename = "mixture.txt";
	const char * neural_network_toolset::ann_subfolder_name = "batch";
	const char * neural_network_toolset::ann_resume_subfolder_name = "resume";
	const char * neural_network_toolset::trained_ann_index_extractor_pattern = "^ann_trained_(\\d+)\\.data$";
	const char * neural_network_toolset::resume_ann_index_extractor_pattern = "^ann_trained_(\\d+)_epoch_(\\d+)\\.data$";
	const char * neural_network_toolset::output_neurons_extractor_pattern = "output_neurons(.*)\\.dt";
	const char * neural_network_toolset::logfile_name = "log.txt";
	float neural_network_toolset::check_gradient_step_modifiers[] = {1.0, sqrtf(10.0F), 1.0F / sqrtf(10.0F), 10.0F, 0.1F, 10.0F * sqrtf(10.0F), 1.0F / (sqrtf(10.0F) * 10.0F), 100.0F, 0.01F, 100.0F * sqrtf(10.0F), 1.0F / (sqrtf(10.0F) * 100.0F), 1000.0F, 0.001F, -1.0F};

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
		else if (!action.compare("check_gradient"))
		{
			check_gradient();
		}
		else
		{
			do_custom_action();
		}
	}

	void neural_network_toolset::prepare_testing_data()
	{
		throw std::runtime_error("This toolset doesn't implement preparing testing data");
	}
	
	void neural_network_toolset::prepare_training_data()
	{
		throw std::runtime_error("This toolset doesn't implement preparing training data");
	}

	network_schema_smart_ptr neural_network_toolset::get_schema() const
	{
		throw std::runtime_error("This toolset doesn't implement get_schema");
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
			("action,A", boost::program_options::value<std::string>(&action)->default_value(get_default_action()), "run action (info, create, prepare_training_data, prepare_testing_data, randomize_data, generate_input_normalizer, generate_output_normalizer, test, test_batch, validate, validate_batch, validate_infinite, train, snapshot, snapshot_data, snapshot_invalid, ann_snapshot, profile_updater, check_gradient)")
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
			("snapshot_extension", boost::program_options::value<std::string>(&snapshot_extension)->default_value("jpg"), "Extension (type) of the files for neuron values snapshots stored as images.")
			("snapshot_extension_video", boost::program_options::value<std::string>(&snapshot_extension_video)->default_value("avi"), "Extension (type) of the files for neuron values snapshots stored as videos.")
			("snapshot_scale", boost::program_options::value<unsigned int>(&snapshot_scale)->default_value(1), "Scale snapshots by this value.")
			("snapshot_video_fps", boost::program_options::value<unsigned int>(&snapshot_video_fps)->default_value(5), "Frames per second when saving video snapshot.")
			("snapshot_ann_index", boost::program_options::value<unsigned int>(&snapshot_ann_index)->default_value(0), "Index of ANN for snapshots.")
			("snapshot_ann_type", boost::program_options::value<std::string>(&snapshot_ann_type)->default_value("image"), "Type of the output for ann data (image, raw).")
			("snapshot_layer_id", boost::program_options::value<int>(&snapshot_layer_id)->default_value(-1), "ID of the layer snapshots should be created for.")
			("learning_rate,L", boost::program_options::value<float>(&learning_rate)->default_value(0.02F), "Global learning rate, Eta/Mu ratio for Stochastic Diagonal Levenberg Marquardt.")
			("learning_rate_decay_tail", boost::program_options::value<unsigned int>(&learning_rate_decay_tail_epoch_count)->default_value(0), "Number of tail iterations with gradually lowering learning rates.")
			("learning_rate_decay_rate", boost::program_options::value<float>(&learning_rate_decay_rate)->default_value(0.5F), "Degradation of learning rate at each tail epoch.")
			("learning_rate_rise_head", boost::program_options::value<unsigned int>(&learning_rate_rise_head_epoch_count)->default_value(0), "Number of head iterations with gradually increasing learning rates.")
			("learning_rate_rise_rate", boost::program_options::value<float>(&learning_rate_rise_rate)->default_value(0.1F), "Increase factor of learning rate at each head epoch (<1.0).")
			("batch_offset", boost::program_options::value<unsigned int>(&batch_offset)->default_value(0), "shift initial ANN ID when batch training.")
			("test_validate_ann_index", boost::program_options::value<int>(&test_validate_ann_index)->default_value(-1), "Index of ANN to test/validate. -1 indicates all ANNs, batch mode.")
			("test_validate_save_output", boost::program_options::value<bool>(&test_validate_save_output)->default_value(false), "Dump output neurons when doing validating/testing.")
			("test_validate_load_output", boost::program_options::value<bool>(&test_validate_load_output)->default_value(false), "Load output neurons when doing validating/testing.")
			("snapshot_data_set", boost::program_options::value<std::string>(&snapshot_data_set)->default_value("training"), "Type of the dataset to use for snapshots (training, validating, testing).")
			("profile_updater_entry_count", boost::program_options::value<unsigned int>(&profile_updater_entry_count)->default_value(1), "The number of entries to process when profiling updater.")
			("check_gradient_weights", boost::program_options::value<std::string>(&check_gradient_weights)->default_value("::"), "The set of weights to check for gradient, in the form Layer:WeightSet:WeightID.")
			("check_gradient_threshold", boost::program_options::value<float>(&check_gradient_threshold)->default_value(1.05F), "Threshold for gradient check.")
			("check_gradient_base_step", boost::program_options::value<float>(&check_gradient_base_step)->default_value(1.0e-3F), "Base step size for gradient check.")
			("training_algo", boost::program_options::value<std::string>(&training_algo)->default_value("sgd"), "Training algorithm (sgd).")
			("dump_resume", boost::program_options::value<bool>(&dump_resume)->default_value(true), "Dump neural network data after each epoch.")
			("load_resume,R", boost::program_options::value<bool>(&load_resume)->default_value(false), "Resume neural network training strating from saved.")
			("epoch_count_in_training_set", boost::program_options::value<unsigned int>(&epoch_count_in_training_set)->default_value(1), "The whole should be split in this amount of epochs.")
			("weight_decay", boost::program_options::value<float>(&weight_decay)->default_value(0.0F), "Weight decay.")
			("batch_size,B", boost::program_options::value<unsigned int>(&batch_size)->default_value(1), "Training mini-batch size.")
			("momentum,M", boost::program_options::value<float>(&momentum)->default_value(0.0F), "Momentum in training.")
			("shuffle_block_size", boost::program_options::value<int>(&shuffle_block_size)->default_value(-1), "The size of contiguous blocks when shuffling training data, -1 indicates no shuffling.")
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
		std::cout << "Forking output log to " << logfile_path.string() << "..." << std::endl;
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
		analyzer_factory = factory->create_analyzer_factory();

		return (action.size() > 0);
	}

	void neural_network_toolset::do_custom_action()
	{
		throw std::runtime_error((boost::format("Unknown action: %1%") % action).str());
	}

	void neural_network_toolset::dump_settings()
	{
		{
			std::cout << "action" << "=" << action << std::endl;
			std::cout << "input_data_folder" << "=" << input_data_folder << std::endl;
			std::cout << "working_data_folder" << "=" << working_data_folder << std::endl;
			std::cout << "ann_count" << "=" << ann_count << std::endl;
			std::cout << "training_epoch_count" << "=" << training_epoch_count << std::endl;
			std::cout << "snapshot_count" << "=" << snapshot_count << std::endl;
			std::cout << "snapshot_extension" << "=" << snapshot_extension << std::endl;
			std::cout << "snapshot_extension_video" << "=" << snapshot_extension_video << std::endl;
			std::cout << "snapshot_scale" << "=" << snapshot_scale << std::endl;
			std::cout << "snapshot_video_fps" << "=" << snapshot_video_fps << std::endl;
			std::cout << "snapshot_ann_index" << "=" << snapshot_ann_index << std::endl;
			std::cout << "snapshot_ann_type" << "=" << snapshot_ann_type << std::endl;
			std::cout << "snapshot_layer_id" << "=" << snapshot_layer_id << std::endl;
			std::cout << "learning_rate" << "=" << learning_rate << std::endl;
			std::cout << "learning_rate_decay_tail" << "=" << learning_rate_decay_tail_epoch_count << std::endl;
			std::cout << "learning_rate_decay_rate" << "=" << learning_rate_decay_rate << std::endl;
			std::cout << "learning_rate_rise_head" << "=" << learning_rate_rise_head_epoch_count << std::endl;
			std::cout << "learning_rate_rise_rate" << "=" << learning_rate_rise_rate << std::endl;
			std::cout << "batch_offset" << "=" << batch_offset << std::endl;
			std::cout << "test_validate_ann_index" << "=" << test_validate_ann_index << std::endl;
			std::cout << "snapshot_data_set" << "=" << snapshot_data_set << std::endl;
			std::cout << "profile_updater_entry_count" << "=" << profile_updater_entry_count << std::endl;
			std::cout << "check_gradient_weights" << "=" << check_gradient_weights << std::endl;
			std::cout << "check_gradient_threshold" << "=" << check_gradient_threshold << std::endl;
			std::cout << "check_gradient_base_step" << "=" << check_gradient_base_step << std::endl;
			std::cout << "training_algo" << "=" << training_algo << std::endl;
			std::cout << "dump_resume" << "=" << dump_resume << std::endl;
			std::cout << "load_resume" << "=" << load_resume << std::endl;
			std::cout << "epoch_count_in_training_set" << "=" << epoch_count_in_training_set << std::endl;
			std::cout << "weight_decay" << "=" << weight_decay << std::endl;
			std::cout << "batch_size" << "=" << batch_size << std::endl;
			std::cout << "momentum" << "=" << momentum << std::endl;
			std::cout << "shuffle_block_size" << "=" << shuffle_block_size << std::endl;
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

	std::string neural_network_toolset::get_default_action() const
	{
		return std::string();
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

	network_trainer_smart_ptr neural_network_toolset::get_network_trainer(network_schema_smart_ptr schema) const
	{
		network_trainer_smart_ptr res;

		network_updater_smart_ptr updater = updater_factory->create(
			schema,
			get_error_function());

		if (training_algo == "sgd")
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
		res->weight_decay = weight_decay;
		res->batch_size = batch_size;
		res->momentum = momentum;

		return res;
	}

	std::pair<unsupervised_data_reader_smart_ptr, unsigned int> neural_network_toolset::get_data_reader_and_sample_count_for_snapshots() const
	{
		if (snapshot_data_set == "training")
			return std::make_pair(get_data_reader_for_training(false, false), 1);
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

		writer->write_randomized(*reader);
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

				tester->clear_data();

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

				tester->clear_data();

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

			output_neuron_value_set_smart_ptr aggr_neuron_value_set;
			if (test_validate_load_output)
			{
				aggr_neuron_value_set = load_output_neuron_value_set();
			}
			else
			{
				std::vector<output_neuron_value_set_smart_ptr> predicted_neuron_value_set_list = run_batch(*reader_and_sample_count.first, actual_neuron_value_set);

				aggr_neuron_value_set = output_neuron_value_set_smart_ptr(new output_neuron_value_set(predicted_neuron_value_set_list, output_neuron_value_set::merge_average));
			}

			if (test_validate_save_output)
				save_output_neuron_value_set(*aggr_neuron_value_set);

			testing_complete_result_set complete_result_set_avg(get_error_function(), actual_neuron_value_set);
			{
				complete_result_set_avg.predicted_output_neuron_value_set = aggr_neuron_value_set;
				complete_result_set_avg.recalculate_mse();
				std::cout << "Merged (average), ";
				get_validating_visualizer()->dump(std::cout, complete_result_set_avg);
				std::cout << std::endl;
			}
		}
		else if (boost::filesystem::exists(get_working_data_folder() / testing_unsupervised_data_filename))
		{
			output_neuron_value_set_smart_ptr aggr_neuron_value_set;
			if (test_validate_load_output)
			{
				aggr_neuron_value_set = load_output_neuron_value_set();
			}
			else
			{
				std::pair<unsupervised_data_reader_smart_ptr, unsigned int> reader_and_sample_count = get_data_reader_for_testing_unsupervised_and_sample_count();

				std::vector<output_neuron_value_set_smart_ptr> predicted_neuron_value_set_list = run_batch(*reader_and_sample_count.first, reader_and_sample_count.second);

				aggr_neuron_value_set = output_neuron_value_set_smart_ptr(new output_neuron_value_set(predicted_neuron_value_set_list, nnforge::output_neuron_value_set::merge_average));
			}

			if (test_validate_save_output)
				save_output_neuron_value_set(*aggr_neuron_value_set);

			run_test_with_unsupervised_data(*aggr_neuron_value_set);
		}
		else throw neural_network_exception((boost::format("File %1% doesn't exist - nothing to test") % (get_working_data_folder() / testing_unsupervised_data_filename).string()).str());
	}

	void neural_network_toolset::save_output_neuron_value_set(const output_neuron_value_set& neuron_value_set) const
	{
		boost::filesystem::path output_folder = get_working_data_folder() / output_subfolder_name;
		boost::filesystem::create_directories(output_folder);
		boost::filesystem::path output_neurons_filepath = output_folder / output_neurons_filename;

		boost::filesystem::ofstream output_neurons_file(output_neurons_filepath, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
		std::cout << "Writing output neurons to " << output_neurons_filepath.string() << " ..." << std::endl;
		neuron_value_set.write(output_neurons_file);
	}

	output_neuron_value_set_smart_ptr neural_network_toolset::load_output_neuron_value_set() const
	{
		boost::filesystem::path output_folder = get_working_data_folder() / output_subfolder_name;
		boost::filesystem::path mixture_filepath = output_folder / mixture_filename;

		std::vector<std::pair<std::string, float> > filename_weight_list;
		if (boost::filesystem::exists(mixture_filepath))
		{
			boost::filesystem::ifstream file_input(mixture_filepath, std::ios_base::in);
			float weight_sum = 0.0F;
			unsigned int no_weight_count = 0;
			while (true)
			{
				std::string str;
				std::getline(file_input, str);
				boost::trim(str);
				if (str.empty())
					break;
				std::vector<std::string> strs;
				boost::split(strs, str, boost::is_any_of("\t"));

				if ((strs.size() == 1) || (strs[1].empty()))
				{
					++no_weight_count;
					filename_weight_list.push_back(std::make_pair(strs[0], -std::numeric_limits<float>::max()));
				}
				else
				{
					float weight = static_cast<float>(atof(strs[1].c_str()));
					weight_sum += weight;
					filename_weight_list.push_back(std::make_pair(strs[0], weight));
				}
			}
			if (no_weight_count > 0)
			{
				float weight = (1.0F - weight_sum) / static_cast<float>(no_weight_count);
				for(std::vector<std::pair<std::string, float> >::iterator it = filename_weight_list.begin(); it != filename_weight_list.end(); ++it)
				{
					if (it->second == -std::numeric_limits<float>::max())
						it->second = weight;
				}
			}
		}
		else
		{
			nnforge_regex expression(output_neurons_extractor_pattern);
			nnforge_cmatch what;

			for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(output_folder); it != boost::filesystem::directory_iterator(); ++it)
			{
				boost::filesystem::path file_path = it->path();
				std::string file_name = file_path.filename().string();

				if (nnforge_regex_search(file_name.c_str(), what, expression))
				{
					filename_weight_list.push_back(std::make_pair(file_name, 0.0F));
				}
			}

			float weight = 1.0F / static_cast<float>(filename_weight_list.size());
			for(std::vector<std::pair<std::string, float> >::iterator it = filename_weight_list.begin(); it != filename_weight_list.end(); ++it)
				it->second = weight;
		}

		std::vector<std::pair<output_neuron_value_set_smart_ptr, float> > output_neuron_value_set_weight_list;
		for(std::vector<std::pair<std::string, float> >::const_iterator it = filename_weight_list.begin(); it != filename_weight_list.end(); ++it)
		{
			std::cout << "Using output neurons from " << it->first << " with weight " << it->second << std::endl;
			boost::filesystem::path output_filepath = output_folder / it->first;
			boost::filesystem::ifstream output_neurons_file(output_filepath, std::ios_base::in | std::ios_base::binary);
			output_neuron_value_set_smart_ptr output_neurons(new output_neuron_value_set());
			output_neurons->read(output_neurons_file);
			output_neuron_value_set_weight_list.push_back(std::make_pair(output_neurons, it->second));
		}

		output_neuron_value_set_smart_ptr aggr_neuron_value_set(new output_neuron_value_set(output_neuron_value_set_weight_list));

		return aggr_neuron_value_set;
	}

	void neural_network_toolset::run_test_with_unsupervised_data(const output_neuron_value_set& neuron_value_set)
	{
		throw neural_network_exception("Running test with unsupervised data is not implemented by the derived toolset");
	}

	void neural_network_toolset::generate_input_normalizer()
	{
		nnforge::supervised_data_reader_smart_ptr reader = get_initial_data_reader_for_normalizing();;

		std::vector<nnforge::feature_map_data_stat> feature_map_data_stat_list = reader->get_feature_map_input_data_stat_list();
		unsigned int feature_map_id = 0;
		for(std::vector<nnforge::feature_map_data_stat>::const_iterator it = feature_map_data_stat_list.begin(); it != feature_map_data_stat_list.end(); ++it, ++feature_map_id)
			std::cout << "Feature map # " << feature_map_id << ": " << *it << std::endl;

		normalize_data_transformer normalizer(feature_map_data_stat_list);

		boost::filesystem::ofstream file_with_schema(get_working_data_folder() / normalizer_input_filename, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
		normalizer.write(file_with_schema);
	}

	void neural_network_toolset::generate_output_normalizer()
	{
		nnforge::supervised_data_reader_smart_ptr reader = get_initial_data_reader_for_normalizing();;

		std::vector<nnforge::feature_map_data_stat> feature_map_data_stat_list = reader->get_feature_map_output_data_stat_list();
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

	normalize_data_transformer_smart_ptr neural_network_toolset::get_reverse_input_data_normalize_transformer() const
	{
		return get_input_data_normalize_transformer()->get_inverted_transformer();
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
						unsigned int offset = static_cast<unsigned int>(max_it - src_it);
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
						unsigned int offset = static_cast<unsigned int>(max_it - src_it);
						float val = *max_it;

						if (val > *max_val_it)
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

			std::vector<unsigned int> snapshot_data_dimension_list = get_snapshot_data_dimension_list(static_cast<unsigned int>(sn.config.dimension_sizes.size()));

			if (snapshot_data_dimension_list.size() == 2)
			{
				boost::filesystem::path snapshot_file_path = snapshot_folder / (boost::format("snapshot_%|1$05d|.%2%") % i % snapshot_extension).str();

				snapshot_visualizer::save_2d_snapshot(
					sn,
					snapshot_file_path.string().c_str(),
					is_rgb_input() && (sn.config.feature_map_count == 3),
					true,
					snapshot_scale,
					snapshot_data_dimension_list);
			}
			else if (snapshot_data_dimension_list.size() == 3)
			{
				boost::filesystem::path snapshot_file_path = snapshot_folder / (boost::format("snapshot_%|1$05d|.%2%") % i % snapshot_extension_video).str();

				snapshot_visualizer::save_3d_snapshot(
					sn,
					snapshot_file_path.string().c_str(),
					is_rgb_input() && (sn.config.feature_map_count == 3),
					true,
					snapshot_video_fps,
					snapshot_scale,
					snapshot_data_dimension_list);
			}
			else
				throw neural_network_exception((boost::format("Saving snapshot for %1% dimensions is not implemented") % snapshot_data_dimension_list.size()).str());
		}
	}

	std::vector<unsigned int> neural_network_toolset::get_snapshot_data_dimension_list(unsigned int original_dimension_count) const
	{
		std::vector<unsigned int> res;
		
		for(unsigned int i = 0; i < original_dimension_count; ++i)
			res.push_back(i);

		return res;
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
			if ((snapshot_layer_id != -1) && (snapshot_layer_id != layer_id))
				continue;

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
					if ((snapshot_layer_id != -1) && (snapshot_layer_id != layer_id))
						continue;

					unsigned int feature_map_id = sample_location_it->feature_map_id;
					unsigned int offset = sample_location_it->offset;

					boost::filesystem::path snapshot_layer_folder = snapshot_folder / (boost::format("%|1$03d|") % layer_id).str();
					boost::filesystem::create_directories(snapshot_layer_folder);

					std::pair<layer_configuration_specific_snapshot_smart_ptr, layer_configuration_specific_snapshot_smart_ptr> input_image_pair = run_analyzer_for_single_neuron(
						*analyzer,
						layer_id,
						feature_map_id,
						layer_config_list[layer_id + 1].get_offsets(offset),
						layer_config_list[layer_id + 1].feature_map_count);

					std::vector<unsigned int> snapshot_data_dimension_list = get_snapshot_data_dimension_list(static_cast<unsigned int>(input_image_pair.first->config.dimension_sizes.size()));

					if (snapshot_data_dimension_list.size() == 2)
					{
						boost::filesystem::path snapshot_file_path = snapshot_layer_folder / ((boost::format("snapshot_layer_%|1$03d|_fm_%|2$04d|_%3%_sample_%4%") % layer_id % feature_map_id % snapshot_data_set % current_sample_id).str() + "." + snapshot_extension);
						boost::filesystem::path original_snapshot_file_path = snapshot_layer_folder / ((boost::format("snapshot_layer_%|1$03d|_fm_%|2$04d|_%3%_sample_%4%_original") % layer_id % feature_map_id % snapshot_data_set % current_sample_id).str() + "." + snapshot_extension);

						snapshot_visualizer::save_2d_snapshot(
							*(input_image_pair.first),
							snapshot_file_path.string().c_str(),
							is_rgb_input() && (input_image_pair.first->config.feature_map_count == 3),
							true,
							snapshot_scale,
							snapshot_data_dimension_list);

						if (should_apply_data_transform_to_input_when_visualizing())
						{
							get_reverse_input_data_normalize_transformer()->transform(0, &input_image_pair.second->data[0], neuron_data_type::type_float, input_image_pair.second->config, 0);
						}

						snapshot_visualizer::save_2d_snapshot(
							*(input_image_pair.second),
							original_snapshot_file_path.string().c_str(),
							is_rgb_input() && (input_image_pair.second->config.feature_map_count == 3),
							false,
							snapshot_scale,
							snapshot_data_dimension_list);
					}
					else
						throw neural_network_exception((boost::format("Saving snapshot for %1% dimensions is not implemented") % snapshot_data_dimension_list.size()).str());
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
		network_data_smart_ptr data = load_ann_data(snapshot_ann_index);

		if (snapshot_ann_type == "image")
		{
			std::string ann_snapshot_filename = "ann_snapshot_trained." + snapshot_extension;
			std::vector<layer_data_configuration_list> layer_data_configuration_list_list = schema->get_layer_data_configuration_list_list();
			save_ann_snapshot_image(ann_snapshot_filename, *data, layer_data_configuration_list_list);
		}
		else if (snapshot_ann_type == "raw")
		{
			save_ann_snapshot_raw("ann_snapshot_trained", *data);
		}
		else
			throw std::runtime_error((boost::format("Invalid snapshot_ann_type: %1%") % snapshot_ann_type).str());
	}

	void neural_network_toolset::save_ann_snapshot_raw(
		const std::string& filename_prefix,
		const network_data& data)
	{
		boost::filesystem::path ann_snapshot_folder = get_working_data_folder() / ann_snapshot_subfolder_name;
		boost::filesystem::create_directories(ann_snapshot_folder);

		for(int layer_id = 0; layer_id < data.data_list.size(); ++layer_id)
		{
			for(int part_id = 0; part_id < data.data_list[layer_id]->size(); ++part_id)
			{
				std::string filename = (boost::format("%1%_%|2$02d|_%|3$02d|.txt") % filename_prefix % layer_id % part_id).str();
				boost::filesystem::path ann_snapshot_file_path = ann_snapshot_folder / filename;
				
				debug_util::dump_list(
					&(*data.data_list[layer_id]->at(part_id).begin()),
					data.data_list[layer_id]->at(part_id).size(),
					ann_snapshot_file_path.string().c_str());
			}
		}

		for(int layer_id = 0; layer_id < data.data_custom_list.size(); ++layer_id)
		{
			for(int part_id = 0; part_id < data.data_custom_list[layer_id]->size(); ++part_id)
			{
				std::string filename = (boost::format("%1%_%|2$02d|_custom_%|3$02d|.txt") % filename_prefix % layer_id % part_id).str();
				boost::filesystem::path ann_snapshot_file_path = ann_snapshot_folder / filename;
				
				debug_util::dump_list(
					&(*data.data_custom_list[layer_id]->at(part_id).begin()),
					data.data_custom_list[layer_id]->at(part_id).size(),
					ann_snapshot_file_path.string().c_str());
			}
		}
	}

	void neural_network_toolset::save_ann_snapshot_image(
		const std::string& filename,
		const network_data& data,
		const std::vector<layer_data_configuration_list>& layer_data_configuration_list_list)
	{
		boost::filesystem::path ann_snapshot_folder = get_working_data_folder() / ann_snapshot_subfolder_name;
		boost::filesystem::create_directories(ann_snapshot_folder);

		boost::filesystem::path ann_snapshot_file_path = ann_snapshot_folder / filename;

		snapshot_visualizer::save_ann_snapshot(data.data_list, layer_data_configuration_list_list, ann_snapshot_file_path.string().c_str());
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

	supervised_data_reader_smart_ptr neural_network_toolset::get_data_reader_for_training(
		bool deterministic_transformers_only,
		bool shuffle_entries) const
	{
		supervised_data_reader_smart_ptr current_reader = get_initial_data_reader_for_training(deterministic_transformers_only);

		if (shuffle_entries && (shuffle_block_size > 0))
		{
			supervised_data_reader_smart_ptr new_reader(new supervised_shuffle_entries_data_reader(current_reader, shuffle_block_size));
			current_reader = new_reader;
		}

		if (epoch_count_in_training_set > 1)
		{
			supervised_data_reader_smart_ptr new_reader(new supervised_multiple_epoch_data_reader(current_reader, epoch_count_in_training_set));
			current_reader = new_reader;
		}

		{
			std::vector<data_transformer_smart_ptr> data_transformer_list = get_input_data_transformer_list_for_training();
			for(std::vector<data_transformer_smart_ptr>::iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
			{
				if ((!deterministic_transformers_only) || (*it)->is_deterministic())
				{
					supervised_data_reader_smart_ptr new_reader(new supervised_transformed_input_data_reader(current_reader, *it));
					current_reader = new_reader;
				}
			}
		}
		{
			std::vector<data_transformer_smart_ptr> data_transformer_list = get_output_data_transformer_list_for_training();
			for(std::vector<data_transformer_smart_ptr>::iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
			{
				if ((!deterministic_transformers_only) || (*it)->is_deterministic())
				{
					supervised_data_reader_smart_ptr new_reader(new supervised_transformed_output_data_reader(current_reader, *it));
					current_reader = new_reader;
				}
			}
		}
		return current_reader;
	}

	supervised_data_reader_smart_ptr neural_network_toolset::get_initial_data_reader_for_training(bool force_deterministic) const
	{
		nnforge_shared_ptr<std::istream> training_data_stream(new boost::filesystem::ifstream(get_working_data_folder() / training_randomized_data_filename, std::ios_base::in | std::ios_base::binary));
		supervised_data_reader_smart_ptr current_reader(new supervised_data_stream_reader(training_data_stream));
		return current_reader;
	}

	supervised_data_reader_smart_ptr neural_network_toolset::get_initial_data_reader_for_normalizing() const
	{
		nnforge_shared_ptr<std::istream> training_data_stream(new boost::filesystem::ifstream(get_working_data_folder() / training_data_filename, std::ios_base::in | std::ios_base::binary));
		supervised_data_reader_smart_ptr current_reader(new supervised_data_stream_reader(training_data_stream));
		return current_reader;
	}

	std::pair<supervised_data_reader_smart_ptr, unsigned int> neural_network_toolset::get_data_reader_for_validating_and_sample_count() const
	{
		supervised_data_reader_smart_ptr current_reader = get_initial_data_reader_for_validating();

		{
			std::vector<data_transformer_smart_ptr> data_transformer_list = get_input_data_transformer_list_for_validating();
			for(std::vector<data_transformer_smart_ptr>::iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
			{
				supervised_data_reader_smart_ptr new_reader(new supervised_transformed_input_data_reader(current_reader, *it));
				current_reader = new_reader;
			}
		}
		{
			std::vector<data_transformer_smart_ptr> data_transformer_list = get_output_data_transformer_list_for_validating();
			for(std::vector<data_transformer_smart_ptr>::iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
			{
				supervised_data_reader_smart_ptr new_reader(new supervised_transformed_output_data_reader(current_reader, *it));
				current_reader = new_reader;
			}
		}

		return std::make_pair(current_reader, get_validating_sample_count() * current_reader->get_sample_count());
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
		{
			std::vector<data_transformer_smart_ptr> data_transformer_list = get_input_data_transformer_list_for_testing();
			for(std::vector<data_transformer_smart_ptr>::iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
			{
				supervised_data_reader_smart_ptr new_reader(new supervised_transformed_input_data_reader(current_reader, *it));
				current_reader = new_reader;
			}
		}
		{
			std::vector<data_transformer_smart_ptr> data_transformer_list = get_output_data_transformer_list_for_testing();
			for(std::vector<data_transformer_smart_ptr>::iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
			{
				supervised_data_reader_smart_ptr new_reader(new supervised_transformed_output_data_reader(current_reader, *it));
				current_reader = new_reader;
			}
		}

		return std::make_pair(current_reader, get_testing_sample_count() * current_reader->get_sample_count());
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
		{
			std::vector<data_transformer_smart_ptr> data_transformer_list = get_input_data_transformer_list_for_testing();
			for(std::vector<data_transformer_smart_ptr>::iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
			{
				unsupervised_data_reader_smart_ptr new_reader(new unsupervised_transformed_input_data_reader(current_reader, *it));
				current_reader = new_reader;
			}
		}

		return std::make_pair(current_reader, get_testing_sample_count() * current_reader->get_sample_count());
	}

	unsupervised_data_reader_smart_ptr neural_network_toolset::get_initial_data_reader_for_testing_unsupervised() const
	{
		nnforge_shared_ptr<std::istream> testing_data_stream(new boost::filesystem::ifstream(get_working_data_folder() / testing_unsupervised_data_filename, std::ios_base::in | std::ios_base::binary));
		unsupervised_data_reader_smart_ptr current_reader(new unsupervised_data_stream_reader(testing_data_stream));
		return current_reader;
	}

	std::map<unsigned int, unsigned int> neural_network_toolset::get_resume_ann_list(const std::set<unsigned int>& exclusion_ann_list) const
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

	std::set<unsigned int> neural_network_toolset::get_trained_ann_list() const
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

	std::vector<network_data_peek_entry> neural_network_toolset::get_resume_ann_list_entry_list() const
	{
		std::vector<network_data_peek_entry> res;

		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();
		boost::filesystem::create_directories(batch_folder);
		boost::filesystem::path resume_ann_folder_path = batch_folder / ann_resume_subfolder_name;
		boost::filesystem::create_directories(resume_ann_folder_path);

		std::set<unsigned int> trained_ann_list = get_trained_ann_list();

		std::map<unsigned int, unsigned int> resume_ann_list = get_resume_ann_list(trained_ann_list);

		for(std::map<unsigned int, unsigned int>::const_iterator it = resume_ann_list.begin(); it != resume_ann_list.end(); ++it)
		{
			network_data_peek_entry new_item;
			new_item.index = it->first;
			new_item.start_epoch = it->second;
			std::string filename = (boost::format("ann_trained_%|1$03d|_epoch_%|2$05d|.data") % new_item.index % new_item.start_epoch).str();
			boost::filesystem::path filepath = resume_ann_folder_path / filename;
			new_item.data = network_data_smart_ptr(new network_data());
			{
				boost::filesystem::ifstream in(filepath, std::ios_base::in | std::ios_base::binary);
				new_item.data->read(in);
			}

			res.push_back(new_item);
		}

		std::sort(res.begin(), res.end(), compare_entry);

		return res;
	}

	bool neural_network_toolset::compare_entry(network_data_peek_entry i, network_data_peek_entry j)
	{
		return (i.index > j.index);
	}

	void neural_network_toolset::train()
	{
		network_schema_smart_ptr schema(new network_schema());
		{
			boost::filesystem::ifstream in(get_working_data_folder() / schema_filename, std::ios_base::in | std::ios_base::binary);
			schema->read(in);
		}

		network_trainer_smart_ptr trainer = get_network_trainer(schema);

		supervised_data_reader_smart_ptr training_data_reader = get_data_reader_for_training(false, true);

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
		nnforge_shared_ptr<network_data_peeker> peeker = nnforge_shared_ptr<network_data_peeker>(new network_data_peeker_random(get_network_output_type(), ann_count, starting_index, leading_tasks));

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
			get_error_function());

		supervised_data_reader_smart_ptr training_data_reader = get_data_reader_for_training(true, false);
		training_data_reader = supervised_data_reader_smart_ptr(new supervised_limited_entry_count_data_reader(training_data_reader, profile_updater_entry_count));

		network_data_smart_ptr data(new network_data(*schema));
		{
			random_generator data_gen = rnd::get_random_generator(47597);
			data->randomize(*schema, data_gen);
			network_data_initializer().initialize(
				data->data_list,
				*schema,
				get_network_output_type());
		}

		std::vector<std::vector<float> > learning_rates;
		for(layer_data_list::const_iterator it = data->data_list.begin(); it != data->data_list.end(); ++it)
			learning_rates.push_back(std::vector<float>((*it)->size(), learning_rate));
		/*
		{
			boost::filesystem::ofstream data_file(get_working_data_folder() / ann_subfolder_name / "ann_trained_000_profile_updater_init.data", std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
			data->write(data_file);
		}
		*/
		boost::chrono::steady_clock::time_point start = boost::chrono::high_resolution_clock::now();
		std::pair<testing_result_smart_ptr, training_stat_smart_ptr> training_result = updater->update(
			*training_data_reader,
			learning_rates,
			data,
			batch_size,
			weight_decay,
			momentum,
			true);
		boost::chrono::duration<float> sec = boost::chrono::high_resolution_clock::now() - start;
		/*
		{
			boost::filesystem::ofstream data_file(get_working_data_folder() / ann_subfolder_name / "ann_trained_000_profile_updater.data", std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
			data->write(data_file);
		}
		*/
		// save_ann_snapshot_raw("ann_snapshot_profile_updater", *data);

		float time_to_complete_seconds = sec.count();

		if (time_to_complete_seconds != 0.0F)
		{
			float flops = static_cast<float>(training_data_reader->get_entry_count()) * updater->get_flops_for_single_entry() * static_cast<float>(ann_count);
			float gflops = flops / time_to_complete_seconds * 1.0e-9F;
			std::cout << (boost::format("%|1$.1f| GFLOPs, %|2$.2f| seconds") % gflops % time_to_complete_seconds) << std::endl;
		}

		std::cout << *training_result.second << std::endl;

		std::cout << data->data_list.get_stat() << std::endl;
	}

	void neural_network_toolset::check_gradient()
	{
		network_schema_smart_ptr schema(new network_schema());
		{
			boost::filesystem::ifstream in(get_working_data_folder() / schema_filename, std::ios_base::in | std::ios_base::binary);
			schema->read(in);
		}

		network_updater_smart_ptr updater = updater_factory->create(
			schema,
			get_error_function());

		supervised_data_reader_smart_ptr training_data_reader = get_data_reader_for_training(true, false);
		training_data_reader = supervised_data_reader_smart_ptr(new supervised_limited_entry_count_data_reader(training_data_reader, 1));

		network_data_smart_ptr data(new network_data(*schema));
		{
			random_generator data_gen = rnd::get_random_generator(47597);
			data->randomize(*schema, data_gen);
			network_data_initializer().initialize(
				data->data_list,
				*schema,
				get_network_output_type());
		}

		std::vector<std::vector<float> > learning_rates;
		for(layer_data_list::const_iterator it = data->data_list.begin(); it != data->data_list.end(); ++it)
			learning_rates.push_back(std::vector<float>((*it)->size(), 0.0F));

		std::vector<std::string> check_gradient_weight_params;
		char* end;
		boost::split(check_gradient_weight_params, check_gradient_weights, boost::is_any_of(":"));

		if (check_gradient_weight_params.size() != 3)
			throw std::runtime_error((boost::format("Invalid check_gradient_weights parameter: %1%") % check_gradient_weights).str());

		int param_layer_id = -1;
		if (!check_gradient_weight_params[0].empty())
			param_layer_id = strtol(check_gradient_weight_params[0].c_str(), &end, 10);
		int param_weight_set = -1;
		if (!check_gradient_weight_params[1].empty())
			param_weight_set = strtol(check_gradient_weight_params[1].c_str(), &end, 10);
		int param_weight_id = -1;
		if (!check_gradient_weight_params[2].empty())
			param_weight_id = strtol(check_gradient_weight_params[2].c_str(), &end, 10);

		const const_layer_list& layer_list = *schema;
		unsigned int error_count = 0;
		unsigned int total_weight_count = 0;
		for(int layer_id = ((param_layer_id == -1) ? 0 : param_layer_id); layer_id < ((param_layer_id == -1) ? layer_list.size() : param_layer_id + 1); ++layer_id)
		{
			layer_data_smart_ptr layer_data = data->data_list[layer_id];
			int min_weight_set = (param_weight_set == -1) ? 0 : param_weight_set;
			int max_weight_set = (param_weight_set == -1) ? static_cast<int>(layer_data->size()) : std::min<int>(static_cast<int>(layer_data->size()), param_weight_set + 1);
			for(int weight_set = min_weight_set; weight_set < max_weight_set; ++weight_set)
			{
				std::vector<float>& weight_list = layer_data->at(weight_set);
				std::vector<int> weight_id_list;
				if (param_weight_id != -1)
				{
					if (param_weight_id < weight_list.size())
						weight_id_list.push_back(param_weight_id);
				}
				else
				{
					for(int i = 0; i < weight_list.size(); ++i)
						weight_id_list.push_back(i);
				}

				random_generator weight_gen = rnd::get_random_generator(637463);
				for(int weight_to_process_count = static_cast<int>(weight_id_list.size()); weight_to_process_count > 0; --weight_to_process_count)
				{
					nnforge_uniform_int_distribution<int> dist(0, weight_to_process_count - 1);
					int index = dist(weight_gen);
					int weight_id = weight_id_list[index];
					int leftover_entry_id = weight_id_list[weight_to_process_count - 1];
					weight_id_list[index] = leftover_entry_id;

					std::cout << layer_id << ":" << weight_set << ":" << weight_id << " ";

					learning_rates[layer_id][weight_set] = 1.0e+6F;
					std::vector<float> original_weights = data->data_list[layer_id]->at(weight_set);
					float original_weight = original_weights[weight_id];

					std::pair<testing_result_smart_ptr, training_stat_smart_ptr> res = updater->update(
						*training_data_reader,
						learning_rates,
						data,
						1,
						0.0F,
						0.0F,
						true);
					double original_error = res.first->get_error();
					float gradient_backprop = -(data->data_list[layer_id]->at(weight_set).at(weight_id) - original_weight) / 1.0e+6F;

					float best_gradient_rate = std::numeric_limits<float>::max();
					float best_check_gradient_step = 0.0F;
					float best_gradient_check = 0.0F;
					for (int step_modifier = 0; (check_gradient_step_modifiers[step_modifier] > 0.0F) && (best_gradient_rate > check_gradient_threshold); ++step_modifier)
					{
						for(int sign = 0; (sign <= 1) && (best_gradient_rate > check_gradient_threshold); ++sign)
						{
							float check_gradient_step = check_gradient_base_step * check_gradient_step_modifiers[step_modifier];
							if (sign == 1)
								check_gradient_step = -check_gradient_step;

							data->data_list[layer_id]->at(weight_set) = original_weights;
							data->data_list[layer_id]->at(weight_set).at(weight_id) = original_weight + check_gradient_step;
							res = updater->update(
								*training_data_reader,
								learning_rates,
								data,
								1,
								0.0F,
								0.0F,
								true);
							float new_error = res.first->get_error();
							float gradient_check = static_cast<float>(new_error - original_error) / check_gradient_step;

							float gradient_rate = get_gradient_rate(gradient_backprop, gradient_check);

							if (gradient_rate <= best_gradient_rate)
							{
								best_gradient_rate = gradient_rate;
								best_check_gradient_step = check_gradient_step;
								best_gradient_check = gradient_check;
							}
						}
					}

					if (best_gradient_rate > check_gradient_threshold)
					{
						std::cout << "ERROR: ";
						++error_count;
					}
					std::cout << "rate=" << best_gradient_rate << ", gradient_backprop=" << gradient_backprop << ", gradient_check=" << best_gradient_check << ", step = " << best_check_gradient_step;
					++total_weight_count;

					data->data_list[layer_id]->at(weight_set) = original_weights;
					learning_rates[layer_id][weight_set] = 0.0F;

					std::cout << std::endl;
				}
			}
		}

		std::cout << error_count << " errors encountered in " << total_weight_count << " weights " << (boost::format("(%|1$.2f|%%)") % (static_cast<float>(error_count) * 100.0F / static_cast<float>(total_weight_count))).str() << std::endl;
	}

	float neural_network_toolset::get_gradient_rate(float gradient_backprop, float gradient_check) const
	{
		if (gradient_backprop == 0.0F)
		{
			if (gradient_check == 0.0F)
				return 1.0F;
			else
				return 1.0e+37F;
		}
		else
		{
			if (gradient_check == 0.0F)
				return 1.0e+37F;

			if (((gradient_backprop > 0.0F) && (gradient_check > 0.0F)) || ((gradient_backprop < 0.0F) && (gradient_check < 0.0F)))
			{
				float diff = std::max(gradient_backprop / gradient_check, gradient_check / gradient_backprop);
				return diff;
			}
			else
			{
				float diff = std::max(std::max(std::max(fabs(gradient_check), fabs(gradient_backprop)), fabs(1.0F / gradient_backprop)), fabs(1.0F / gradient_check));
				return diff;
			}
		}
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

	bool neural_network_toolset::should_apply_data_transform_to_input_when_visualizing() const
	{
		return false;
	}
}
