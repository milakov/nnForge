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

namespace nnforge
{
	const char * neural_network_toolset::training_data_filename = "training.sdt";
	const char * neural_network_toolset::training_randomized_data_filename = "training_randomized.sdt";
	const char * neural_network_toolset::validating_data_filename = "validating.sdt";
	const char * neural_network_toolset::testing_data_filename = "testing.sdt";
	const char * neural_network_toolset::testing_unsupervised_data_filename = "testing.udt";
	const char * neural_network_toolset::schema_filename = "ann.schema";
	const char * neural_network_toolset::data_filename = "ann.data";
	const char * neural_network_toolset::data_trained_filename = "ann_trained.data";
	const char * neural_network_toolset::normalizer_input_filename = "normalizer_input.data";
	const char * neural_network_toolset::normalizer_output_filename = "normalizer_output.data";
	const char * neural_network_toolset::snapshot_subfolder_name = "snapshot";
	const char * neural_network_toolset::ann_snapshot_subfolder_name = "ann_snapshot";
	const char * neural_network_toolset::snapshot_invalid_subfolder_name = "invalid";
	const char * neural_network_toolset::ann_subfolder_name = "batch";
	const char * neural_network_toolset::trained_ann_index_extractor_pattern = "^ann_trained_(\\d+)\\.data$";

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
			validate(true, false);
		}
		else if (!action.compare("test"))
		{
			validate(false, false);
		}
		else if (!action.compare("validate_batch"))
		{
			validate_batch(true);
		}
		else if (!action.compare("test_batch"))
		{
			validate_batch(false);
		}
		else if (!action.compare("validate_infinite"))
		{
			validate(true, true);
		}
		else if (!action.compare("info"))
		{
			factory->info();
		}
		else if (!action.compare("train_batch"))
		{
			train(true);
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
			("action,A", boost::program_options::value<std::string>(&action), "run action (info, create, prepare_training_data, prepare_testing_data, randomize_data, generate_input_normalizer, generate_output_normalizer, test, test_batch, validate, validate_batch, validate_infinite, train, train_batch, snapshot, snapshot_invalid, ann_snapshot, profile_updater, profile_hessian)")
			("config,C", boost::program_options::value<boost::filesystem::path>(&config_file)->default_value(default_config_path), "path to the configuration file.")
			;

		// Declare a group of options that will be 
		// allowed both on command line and in
		// config file
		boost::program_options::options_description config("Configuration");
		config.add_options()
			("input_data_folder,I", boost::program_options::value<boost::filesystem::path>(&input_data_folder)->default_value(""), "path to the folder where input data are located.")
			("working_data_folder,W", boost::program_options::value<boost::filesystem::path>(&working_data_folder)->default_value(""), "path to the folder where data are processed.")
			("ann_count,N", boost::program_options::value<unsigned int>(&ann_count)->default_value(1), "amount of networks to be processed.")
			("training_iteration_count,T", boost::program_options::value<unsigned int>(&training_iteration_count)->default_value(50), "amount of iterations to perform during single ANN training.")
			("snapshot_count", boost::program_options::value<unsigned int>(&snapshot_count)->default_value(100), "amount of snapshots to generate.")
			("snapshot_extension", boost::program_options::value<std::string>(&snapshot_extension)->default_value("jpg"), "Extension (type) of the files for neuron values snapshots.")
			("snapshot_mode", boost::program_options::value<std::string>(&snapshot_mode)->default_value("image"), "Type of the neuron values snapshot to generate (image, video).")
			("snapshot_video_fps", boost::program_options::value<unsigned int>(&snapshot_video_fps)->default_value(5), "Frames per second when saving video snapshot.")
			("ann_snapshot_extension", boost::program_options::value<std::string>(&ann_snapshot_extension)->default_value("jpg"), "Extension (type) of the files for network weights snapshots.")
			("mu_increase_factor", boost::program_options::value<float>(&mu_increase_factor)->default_value(1.3F), "Mu increases by this ratio each iteration.")
			("max_mu", boost::program_options::value<float>(&max_mu)->default_value(5.0e-4F), "Maximum Mu during training.")
			("training_speed", boost::program_options::value<float>(&training_speed)->default_value(0.02F), "Eta/Mu ratio.")
			("training_speed_degradation", boost::program_options::value<float>(&training_speed_degradaton)->default_value(1.0F), "Degradation of training speed at each iteration.")
			("batch_offset", boost::program_options::value<unsigned int>(&batch_offset)->default_value(0), "shift initial ANN ID when batch training.")
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

		factory->initialize();

		tester_factory = factory->create_tester_factory();
		updater_factory = factory->create_updater_factory();
		hessian_factory = factory->create_hessian_factory();

		return (action.size() > 0);
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

	void neural_network_toolset::randomize_data()
	{
		std::tr1::shared_ptr<std::istream> in(new boost::filesystem::ifstream(get_working_data_folder() / training_data_filename, std::ios_base::in | std::ios_base::binary));
		std::tr1::shared_ptr<std::ostream> out(new boost::filesystem::ofstream(get_working_data_folder() / training_randomized_data_filename, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc));

		supervised_data_stream_reader reader(in);

		std::cout << "Randomizing " << reader.get_entry_count() << " entries" << std::endl;

		switch(get_network_output_type())
		{
		case network_output_type::type_classifier:
		case network_output_type::type_roc:
			reader.write_randomized_classifier(out);
			break;
		default:
			reader.write_randomized(out);
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

		network_data data(*schema);

		random_generator gen = rnd::get_random_generator();
		data.randomize(
			*schema,
			gen);

		{
			boost::filesystem::ofstream file_with_data(get_working_data_folder() / data_filename, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
			data.write(file_with_data);
		}

		data.check_network_data_consistency(*schema);
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

	void neural_network_toolset::validate(
		bool is_validate,
		bool infinite)
	{
		network_tester_smart_ptr tester = get_tester();

		network_data_smart_ptr data(new network_data());
		{
			boost::filesystem::ifstream in(get_working_data_folder() / data_trained_filename, std::ios_base::in | std::ios_base::binary);
			data->read(in);
		}

		tester->set_data(data);

		supervised_data_reader_smart_ptr reader = is_validate ? get_data_reader_for_validating() : get_data_reader_for_testing_supervised();

		unsigned int sample_count = is_validate ? get_validating_sample_count() : get_testing_sample_count();
		output_neuron_value_set_smart_ptr actual_neuron_value_set = reader->get_output_neuron_value_set(sample_count);

		do
		{
			testing_complete_result_set testing_res(is_squared_hinge_loss(), actual_neuron_value_set);
			boost::chrono::steady_clock::time_point start = boost::chrono::high_resolution_clock::now();
			tester->test(
				*reader,
				testing_res);
			boost::chrono::duration<float> sec = boost::chrono::high_resolution_clock::now() - start;
			get_validating_visualizer()->dump(std::cout, testing_res);
			std::cout << std::endl;
		}
		while (infinite);
	}

	std::vector<output_neuron_value_set_smart_ptr> neural_network_toolset::run_batch(
		supervised_data_reader& reader,
		output_neuron_value_set_smart_ptr actual_neuron_value_set)
	{
		network_tester_smart_ptr tester = get_tester();

		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();

		std::tr1::regex expression(trained_ann_index_extractor_pattern);
		std::tr1::cmatch what;

		std::vector<float> invalid_ratio_list;
		std::vector<output_neuron_value_set_smart_ptr> predicted_neuron_value_set_list;
		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(batch_folder); it != boost::filesystem::directory_iterator(); ++it)
		{
			boost::filesystem::path file_path = it->path();
			std::string file_name = file_path.filename().string();

			if (std::tr1::regex_search(file_name.c_str(), what, expression))
			{
				unsigned int index = static_cast<unsigned int>(atol(std::string(what[1].first, what[1].second).c_str()));
				network_data_smart_ptr data(new network_data());
				{
					boost::filesystem::ifstream in(file_path, std::ios_base::in | std::ios_base::binary);
					data->read(in);
				}

				tester->set_data(data);

				testing_complete_result_set testing_res(is_squared_hinge_loss(), actual_neuron_value_set);
				boost::chrono::steady_clock::time_point start = boost::chrono::high_resolution_clock::now();
				tester->test(
					reader,
					testing_res);
				boost::chrono::duration<float> sec = boost::chrono::high_resolution_clock::now() - start;
				std::cout << "# " << index << ", ";
				get_validating_visualizer()->dump(std::cout, testing_res);
				std::cout << std::endl;

				predicted_neuron_value_set_list.push_back(testing_res.predicted_output_neuron_value_set);
			}
		}

		return predicted_neuron_value_set_list;
	}

	std::vector<output_neuron_value_set_smart_ptr> neural_network_toolset::run_batch(unsupervised_data_reader& reader)
	{
		network_tester_smart_ptr tester = get_tester();

		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();

		std::tr1::regex expression(trained_ann_index_extractor_pattern);
		std::tr1::cmatch what;

		std::vector<float> invalid_ratio_list;
		std::vector<output_neuron_value_set_smart_ptr> predicted_neuron_value_set_list;
		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(batch_folder); it != boost::filesystem::directory_iterator(); ++it)
		{
			boost::filesystem::path file_path = it->path();
			std::string file_name = file_path.filename().string();

			if (std::tr1::regex_search(file_name.c_str(), what, expression))
			{
				unsigned int index = static_cast<unsigned int>(atol(std::string(what[1].first, what[1].second).c_str()));
				network_data_smart_ptr data(new network_data());
				{
					boost::filesystem::ifstream in(file_path, std::ios_base::in | std::ios_base::binary);
					data->read(in);
				}

				tester->set_data(data);

				boost::chrono::steady_clock::time_point start = boost::chrono::high_resolution_clock::now();
				output_neuron_value_set_smart_ptr new_res = tester->run(reader, get_testing_sample_count());
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

	void neural_network_toolset::validate_batch(bool is_validate)
	{
		if (is_validate || boost::filesystem::exists(get_working_data_folder() / testing_data_filename))
		{
			supervised_data_reader_smart_ptr reader = is_validate ? get_data_reader_for_validating() : get_data_reader_for_testing_supervised();

			unsigned int sample_count = is_validate ? get_validating_sample_count() : get_testing_sample_count();
			output_neuron_value_set_smart_ptr actual_neuron_value_set = reader->get_output_neuron_value_set(sample_count);

			std::vector<output_neuron_value_set_smart_ptr> predicted_neuron_value_set_list = run_batch(*reader, actual_neuron_value_set);

			testing_complete_result_set complete_result_set_avg(is_squared_hinge_loss(), actual_neuron_value_set);
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
			unsupervised_data_reader_smart_ptr reader = get_data_reader_for_testing_unsupervised();

			std::vector<output_neuron_value_set_smart_ptr> predicted_neuron_value_set_list = run_batch(*reader);

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
		std::tr1::shared_ptr<std::istream> in(new boost::filesystem::ifstream(get_working_data_folder() / training_data_filename, std::ios_base::in | std::ios_base::binary));
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
		std::tr1::shared_ptr<std::istream> in(new boost::filesystem::ifstream(get_working_data_folder() / training_data_filename, std::ios_base::in | std::ios_base::binary));
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

	void neural_network_toolset::snapshot()
	{
		network_tester_smart_ptr tester = get_tester();

		network_data_smart_ptr data(new network_data());
		{
			boost::filesystem::ifstream in(get_working_data_folder() / data_trained_filename, std::ios_base::in | std::ios_base::binary);
			data->read(in);
		}

		tester->set_data(data);

		supervised_data_reader_smart_ptr reader = get_data_reader_for_training();

		reader->reset();

		tester->set_input_configuration_specific(reader->get_input_configuration());

		unsigned int image_count = std::min<unsigned int>(snapshot_count, reader->get_entry_count());
		std::vector<unsigned char> input(reader->get_input_configuration().get_neuron_count() * reader->get_input_neuron_elem_size());
		for(unsigned int image_id = 0; image_id < image_count; ++image_id)
		{
			if (!reader->read(&(*input.begin()), 0))
				throw std::runtime_error((boost::format("Only %1% entries available while %2% snapshots requested") % image_id % image_count).str().c_str());

			std::string snapshot_filename = (boost::format("%|1$03d|") % image_id).str();

			std::vector<layer_configuration_specific_snapshot_smart_ptr> data_res = tester->get_snapshot(
				&(*input.begin()),
				reader->get_input_type(),
				reader->get_input_configuration().get_neuron_count());

			save_snapshot(snapshot_filename, data_res);
		}
	}

	void neural_network_toolset::ann_snapshot()
	{
		network_schema_smart_ptr schema(new network_schema());
		{
			boost::filesystem::ifstream in(get_working_data_folder() / schema_filename, std::ios_base::in | std::ios_base::binary);
			schema->read(in);
		}
		std::vector<layer_data_configuration_list> layer_data_configuration_list_list = schema->get_layer_data_configuration_list_list();

		network_data_smart_ptr data(new network_data());
		{
			boost::filesystem::ifstream in(get_working_data_folder() / data_trained_filename, std::ios_base::in | std::ios_base::binary);
			data->read(in);
		}

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

		boost::filesystem::path ann_snapshot_file_path = ann_snapshot_folder / (std::string("ann_snapshot_") + name + "." + ann_snapshot_extension);

		snapshot_visualizer::save_ann_snapshot(data, layer_data_configuration_list_list, ann_snapshot_file_path.string().c_str());
	}

	void neural_network_toolset::save_snapshot(
		const std::string& name,
		const std::vector<layer_configuration_specific_snapshot_smart_ptr>& data,
		bool folder_for_invalid)
	{
		boost::filesystem::path snapshot_folder = get_working_data_folder() / snapshot_subfolder_name;
		if (folder_for_invalid)
			snapshot_folder /= snapshot_invalid_subfolder_name;
		boost::filesystem::create_directories(snapshot_folder);

		boost::filesystem::path snapshot_file_path = snapshot_folder / (std::string("snapshot_") + name + "." + snapshot_extension);
		if (snapshot_mode == "video")
		{
			snapshot_visualizer::save_snapshot_video(
				data,
				snapshot_file_path.string().c_str(),
				snapshot_video_fps);
		}
		else
		{
			snapshot_visualizer::save_snapshot(
				data,
				snapshot_file_path.string().c_str());
		}
	}

	void neural_network_toolset::snapshot_invalid()
	{
		network_tester_smart_ptr tester = get_tester();

		network_data_smart_ptr data(new network_data());
		{
			boost::filesystem::ifstream in(get_working_data_folder() / data_trained_filename, std::ios_base::in | std::ios_base::binary);
			data->read(in);
		}

		tester->set_data(data);

		supervised_data_reader_smart_ptr reader = get_data_reader_for_validating();

		unsigned int sample_count = get_validating_sample_count();
		output_neuron_value_set_smart_ptr actual_neuron_value_set = reader->get_output_neuron_value_set(sample_count);

		testing_complete_result_set testing_res(is_squared_hinge_loss(), actual_neuron_value_set);
		if (reader->get_output_configuration().get_neuron_count() == 1)
			throw "Invalid snapshots is not implemented for single output neuron configuration";
		tester->test(
			*reader,
			testing_res);

		output_neuron_class_set predicted_cs(*testing_res.predicted_output_neuron_value_set);
		output_neuron_class_set actual_cs(*testing_res.actual_output_neuron_value_set);
		classifier_result cr(predicted_cs, actual_cs);

		reader->reset();

		tester->set_input_configuration_specific(reader->get_input_configuration());

		std::vector<unsigned char> input(reader->get_input_configuration().get_neuron_count() * reader->get_input_neuron_elem_size());
		unsigned int entry_id = 0;
		for(std::vector<std::pair<unsigned int, unsigned int> >::const_iterator it = cr.predicted_and_actual_class_pair_id_list.begin();
			it != cr.predicted_and_actual_class_pair_id_list.end();
			it++)
		{
			if (!reader->read(&(*input.begin()), 0))
				throw std::runtime_error("Not enough entries");
			for(unsigned int i = 1; i < sample_count; ++i)
				if (!reader->read(0, 0))
					throw std::runtime_error("Not enough entries");

			const std::pair<unsigned int, unsigned int>& single_result = *it;

			unsigned int predicted_class_id = single_result.first;
			unsigned int actual_class_id = single_result.second;

			if (single_result.first != single_result.second)
			{
				std::string snapshot_filename = (boost::format("actual_%|1$s|_predicted_%|2$s|_entry_%|3$03d|") %
					get_class_name_by_id(actual_class_id) % get_class_name_by_id(predicted_class_id) % entry_id).str();

				std::vector<layer_configuration_specific_snapshot_smart_ptr> res = tester->get_snapshot(
					&(*input.begin()),
					reader->get_input_type(),
					reader->get_input_configuration().get_neuron_count());

				save_snapshot(snapshot_filename, res, true);
			}

			entry_id++;
		}
	}

	unsigned int neural_network_toolset::get_starting_index_for_batch_training()
	{
		std::tr1::regex expression(trained_ann_index_extractor_pattern);
		std::tr1::cmatch what;

		int max_index = -1;
		boost::filesystem::path batch_folder = get_working_data_folder() / get_ann_subfolder_name();
		for(boost::filesystem::directory_iterator it = boost::filesystem::directory_iterator(batch_folder); it != boost::filesystem::directory_iterator(); it++)
		{
			boost::filesystem::path file_path = it->path();
			std::string file_name = file_path.filename().string();

			if (std::tr1::regex_search(file_name.c_str(), what, expression))
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
			supervised_data_reader_smart_ptr validating_data_reader = get_data_reader_for_validating();
			res.push_back(network_data_pusher_smart_ptr(new validate_progress_network_data_pusher(
				tester_factory->create(schema),
				validating_data_reader,
				get_validating_visualizer(),
				is_squared_hinge_loss(),
				get_validating_sample_count())));
		}

		return res;
	}

	supervised_data_reader_smart_ptr neural_network_toolset::get_data_reader_for_training() const
	{
		std::tr1::shared_ptr<std::istream> training_data_stream(new boost::filesystem::ifstream(get_working_data_folder() / training_randomized_data_filename, std::ios_base::in | std::ios_base::binary));
		supervised_data_reader_smart_ptr current_reader(new supervised_data_stream_reader(training_data_stream));
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

	supervised_data_reader_smart_ptr neural_network_toolset::get_data_reader_for_validating() const
	{
		std::tr1::shared_ptr<std::istream> validating_data_stream(new boost::filesystem::ifstream(get_working_data_folder() / validating_data_filename, std::ios_base::in | std::ios_base::binary));
		supervised_data_reader_smart_ptr current_reader(new supervised_data_stream_reader(validating_data_stream));
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
		return current_reader;
	}

	supervised_data_reader_smart_ptr neural_network_toolset::get_data_reader_for_testing_supervised() const
	{
		std::tr1::shared_ptr<std::istream> testing_data_stream(new boost::filesystem::ifstream(get_working_data_folder() / testing_data_filename, std::ios_base::in | std::ios_base::binary));
		supervised_data_reader_smart_ptr current_reader(new supervised_data_stream_reader(testing_data_stream));
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
		return current_reader;
	}

	unsupervised_data_reader_smart_ptr neural_network_toolset::get_data_reader_for_testing_unsupervised() const
	{
		std::tr1::shared_ptr<std::istream> testing_data_stream(new boost::filesystem::ifstream(get_working_data_folder() / testing_unsupervised_data_filename, std::ios_base::in | std::ios_base::binary));
		unsupervised_data_reader_smart_ptr current_reader(new unsupervised_data_stream_reader(testing_data_stream));
		{
			std::vector<data_transformer_smart_ptr> data_transformer_list = get_input_data_transformer_list_for_testing();
			for(std::vector<data_transformer_smart_ptr>::iterator it = data_transformer_list.begin(); it != data_transformer_list.end(); ++it)
			{
				unsupervised_data_reader_smart_ptr new_reader(new unsupervised_transformed_input_data_reader(current_reader, *it));
				current_reader = new_reader;
			}
		}
		return current_reader;
	}

	void neural_network_toolset::train(bool batch)
	{
		network_schema_smart_ptr schema(new network_schema());
		{
			boost::filesystem::ifstream in(get_working_data_folder() / schema_filename, std::ios_base::in | std::ios_base::binary);
			schema->read(in);
		}

		hessian_calculator_smart_ptr hessian = hessian_factory->create(schema);

		network_updater_smart_ptr updater = updater_factory->create(
			schema,
			is_squared_hinge_loss(),
			get_dropout_rate_map(),
			get_weight_vector_bound_map());

		network_trainer_sdlm trainer(
			schema,
			hessian,
			updater);
		trainer.iteration_count = training_iteration_count;
		trainer.speed = training_speed;
		trainer.eta_degradation = training_speed_degradaton;
		trainer.max_mu = max_mu;
		trainer.mu_increase_factor = mu_increase_factor;

		supervised_data_reader_smart_ptr training_data_reader = get_data_reader_for_training();

		std::tr1::shared_ptr<network_data_peeker> peeker;
		boost::filesystem::path batch_folder;
		if (batch)
		{
			batch_folder = get_working_data_folder() / get_ann_subfolder_name();
			boost::filesystem::create_directories(batch_folder);

			unsigned int starting_index = get_starting_index_for_batch_training();
			peeker = std::tr1::shared_ptr<network_data_peeker>(new network_data_peeker_random(ann_count, starting_index));
		}
		else
		{
			network_data_smart_ptr data(new network_data(*schema));
			{
				boost::filesystem::ifstream in(get_working_data_folder() / data_filename, std::ios_base::in | std::ios_base::binary);
				data->read(in);
			}

			peeker = std::tr1::shared_ptr<network_data_peeker>(new network_data_peeker_single(data));
		}

		complex_network_data_pusher progress;
		progress.push_back(network_data_pusher_smart_ptr(new report_progress_network_data_pusher()));

		std::vector<network_data_pusher_smart_ptr> validators_for_training = get_validators_for_training(schema);
		progress.insert(progress.end(), validators_for_training.begin(), validators_for_training.end());

		summarize_network_data_pusher res;

		trainer.train(
			*training_data_reader,
			*peeker,
			progress,
			res);

		if (batch)
		{
			res.save_all(batch_folder);
		}
		else
		{
			boost::filesystem::ofstream file_with_data(get_working_data_folder() / data_trained_filename, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
			res.task_state_list[0].data->write(file_with_data);
		}
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
			is_squared_hinge_loss(),
			get_dropout_rate_map(),
			get_weight_vector_bound_map());

		supervised_data_reader_smart_ptr training_data_reader = get_data_reader_for_training();
		training_data_reader->set_max_entries_to_read(200);

		std::vector<network_data_smart_ptr> training_speeds;
		std::vector<network_data_smart_ptr> data;

		random_generator data_gen = rnd::get_random_generator(47597);
		for(unsigned int i = 0; i < ann_count; ++i)
		{
			network_data_smart_ptr ts(new network_data(*schema));
			ts->fill(training_speed);
			training_speeds.push_back(ts);

			network_data_smart_ptr data_elem(new network_data(*schema));
			data_elem->randomize(*schema, data_gen);
			data.push_back(data_elem);
		}

		std::vector<float> random_uniform_list(1 << 10);
		random_generator gen = rnd::get_random_generator();
		std::tr1::uniform_real<float> dist(0.0F, 1.0F);
		for(std::vector<float>::iterator it = random_uniform_list.begin(); it != random_uniform_list.end(); ++it)
			*it = dist(gen);

		boost::chrono::steady_clock::time_point start = boost::chrono::high_resolution_clock::now();
		updater->update(
			*training_data_reader,
			training_speeds,
			data);
		boost::chrono::duration<float> sec = boost::chrono::high_resolution_clock::now() - start;
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
			boost::filesystem::ifstream in(get_working_data_folder() / data_filename, std::ios_base::in | std::ios_base::binary);
			data->read(in);
		}

		unsigned int hessian_entry_to_process_count = std::min<unsigned int>(std::max<unsigned int>(static_cast<unsigned int>(0.05F * training_data_reader->get_entry_count()), 50), training_data_reader->get_entry_count());
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
			return testing_complete_result_set_visualizer_smart_ptr(new testing_complete_result_set_classifier_visualizer());
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

	bool neural_network_toolset::is_squared_hinge_loss() const
	{
		return false;
	}
}
