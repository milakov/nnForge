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

#pragma once

#include "factory_generator.h"
#include "config_options.h"
#include "network_data_pusher.h"
#include "testing_complete_result_set_visualizer.h"
#include "network_output_type.h"
#include "unsupervised_data_reader.h"
#include "output_neuron_value_set.h"
#include "output_neuron_class_set.h"
#include "layer_data_configuration.h"
#include "data_transformer.h"
#include "data_transformer_util.h"
#include "weight_vector_bound.h"
#include "normalize_data_transformer.h"

#include <boost/filesystem.hpp>

namespace nnforge
{
	class neural_network_toolset
	{
	public:
		neural_network_toolset(factory_generator_smart_ptr factory);

		virtual ~neural_network_toolset();

		// Returns true if action is specified
		bool parse(int argc, char* argv[]);

		std::string get_action() const;

		void do_action();

	protected:
		virtual std::vector<string_option> get_string_options();

		virtual std::vector<bool_option> get_bool_options();

		virtual std::vector<float_option> get_float_options();

		virtual std::vector<int_option> get_int_options();

		virtual void prepare_training_data() = 0;

		virtual void prepare_testing_data();

		virtual network_schema_smart_ptr get_schema() const = 0;

		virtual std::map<unsigned int, float> get_dropout_rate_map() const;

		virtual std::map<unsigned int, weight_vector_bound> get_weight_vector_bound_map() const;

		virtual boost::filesystem::path get_input_data_folder() const;

		virtual boost::filesystem::path get_working_data_folder() const;

		virtual std::string get_class_name_by_id(unsigned int class_id) const;

		virtual network_tester_smart_ptr get_tester();

		virtual std::vector<network_data_pusher_smart_ptr> get_validators_for_training(network_schema_smart_ptr schema);

		virtual network_output_type::output_type get_network_output_type() const;

		virtual bool is_training_with_validation() const;

		virtual testing_complete_result_set_visualizer_smart_ptr get_validating_visualizer() const;

		virtual void run_test_with_unsupervised_data(std::vector<output_neuron_value_set_smart_ptr>& predicted_neuron_value_set_list);

		virtual unsigned int get_testing_sample_count() const;

		virtual unsigned int get_validating_sample_count() const;

		virtual std::vector<data_transformer_smart_ptr> get_input_data_transformer_list_for_training() const;

		virtual std::vector<data_transformer_smart_ptr> get_output_data_transformer_list_for_training() const;

		virtual std::vector<data_transformer_smart_ptr> get_input_data_transformer_list_for_validating() const;

		virtual std::vector<data_transformer_smart_ptr> get_output_data_transformer_list_for_validating() const;

		virtual std::vector<data_transformer_smart_ptr> get_input_data_transformer_list_for_testing() const;

		virtual std::vector<data_transformer_smart_ptr> get_output_data_transformer_list_for_testing() const;

		virtual boost::filesystem::path get_ann_subfolder_name() const;

		virtual bool is_squared_hinge_loss() const;

	protected:
		static const char * training_data_filename;
		static const char * training_randomized_data_filename;
		static const char * validating_data_filename;
		static const char * testing_data_filename;
		static const char * testing_unsupervised_data_filename;
		static const char * schema_filename;
		static const char * data_filename;
		static const char * data_trained_filename;
		static const char * normalizer_input_filename;
		static const char * normalizer_output_filename;
		static const char * snapshot_subfolder_name;
		static const char * ann_snapshot_subfolder_name;
		static const char * snapshot_invalid_subfolder_name;
		static const char * ann_subfolder_name;
		static const char * trained_ann_index_extractor_pattern;

		network_tester_factory_smart_ptr tester_factory;
		network_updater_factory_smart_ptr updater_factory;
		hessian_calculator_factory_smart_ptr hessian_factory;

		std::string action;
		std::string snapshot_extension;
		std::string ann_snapshot_extension;
		unsigned int ann_count;
		unsigned int training_iteration_count;
		unsigned int snapshot_count;
		float training_speed;
		float training_speed_degradaton;
		float max_mu;
		float mu_increase_factor;
		unsigned int batch_offset;
		std::string snapshot_mode;
		unsigned int snapshot_video_fps;

	protected:
		std::vector<output_neuron_value_set_smart_ptr> run_batch(
			supervised_data_reader& reader,
			output_neuron_value_set_smart_ptr actual_neuron_value_set);

		std::vector<output_neuron_value_set_smart_ptr> run_batch(unsupervised_data_reader& reader);

		supervised_data_reader_smart_ptr get_data_reader_for_training() const;

		supervised_data_reader_smart_ptr get_data_reader_for_validating() const;

		supervised_data_reader_smart_ptr get_data_reader_for_testing_supervised() const;

		unsupervised_data_reader_smart_ptr get_data_reader_for_testing_unsupervised() const;

		void randomize_data();

		void create();

		void generate_input_normalizer();

		void generate_output_normalizer();

		unsigned int get_starting_index_for_batch_training();

		void validate(
			bool is_validate,
			bool infinite);

		void validate_batch(bool is_validate);

		void snapshot();

		void snapshot_invalid();

		void ann_snapshot();

		void save_snapshot(
			const std::string& name,
			const std::vector<layer_configuration_specific_snapshot_smart_ptr>& data,
			bool folder_for_invalid = false);

		void save_ann_snapshot(
			const std::string& name,
			const network_data& data,
			const std::vector<layer_data_configuration_list>& layer_data_configuration_list_list);

		void train(bool batch = false);

		void profile_updater();

		void profile_hessian();

		normalize_data_transformer_smart_ptr get_input_data_normalize_transformer() const;

		normalize_data_transformer_smart_ptr get_output_data_normalize_transformer() const;

		normalize_data_transformer_smart_ptr get_reverse_output_data_normalize_transformer() const;

	private:
		factory_generator_smart_ptr factory;

		boost::filesystem::path input_data_folder;
		boost::filesystem::path working_data_folder;

	private:
		neural_network_toolset();
		neural_network_toolset(const neural_network_toolset&);
	};
}
