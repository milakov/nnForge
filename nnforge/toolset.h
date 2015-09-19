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
#include "stream_duplicator.h"
#include "network_trainer.h"

#include <vector>
#include <string>

namespace nnforge
{
	class toolset
	{
	public:
		toolset(factory_generator::ptr master_factory);

		virtual ~toolset();

		// Returns true if action is specified
		bool parse(int argc, char* argv[]);

		std::string get_action() const;

		void do_action();

	protected:
		virtual std::string get_default_action() const;

		virtual void do_custom_action();

		virtual std::vector<string_option> get_string_options();

		virtual std::vector<multi_string_option> get_multi_string_options();

		virtual std::vector<path_option> get_path_options();

		virtual std::vector<bool_option> get_bool_options();

		virtual std::vector<float_option> get_float_options();

		virtual std::vector<int_option> get_int_options();

		virtual boost::filesystem::path get_working_data_folder() const;

		virtual network_schema::ptr load_schema() const;

		virtual void run_inference();

		virtual void dump_schema_dot();

		virtual void train();

		virtual structured_data_bunch_reader::ptr get_reader(const std::string& dataset_name) const;

		virtual boost::filesystem::path get_ann_subfolder_name() const;

		virtual network_trainer::ptr get_network_trainer() const;

		virtual std::vector<network_data_pusher::ptr> get_validators_for_training(network_schema::const_ptr schema);

		virtual bool is_training_with_validation() const;

	private:
		void dump_settings();

		std::vector<std::pair<unsigned int, boost::filesystem::path> > get_ann_data_index_and_filepath_list() const;

		std::vector<network_data_peek_entry> get_resume_ann_list_entry_list() const;

		std::set<unsigned int> get_trained_ann_list() const;

		std::map<unsigned int, unsigned int> get_resume_ann_list(const std::set<unsigned int>& exclusion_ann_list) const;

		unsigned int get_starting_index_for_batch_training() const;

		static bool compare_entry(network_data_peek_entry i, network_data_peek_entry j);

	protected:
		factory_generator::ptr master_factory;

		forward_propagation_factory::ptr forward_propagation_factory;
		backward_propagation_factory::ptr backward_propagation_factory;

		nnforge_shared_ptr<stream_duplicator> out_to_log_duplicator;

	protected:
		std::string action;
		boost::filesystem::path config_file_path;
		boost::filesystem::path working_data_folder;
		std::string schema_filename;
		std::vector<std::string> inference_output_layer_names;
		std::string inference_dataset_name;
		std::string training_dataset_name;
		int inference_ann_data_index;
		bool debug_mode;
		std::vector<std::string> training_output_layer_names;
		std::vector<std::string> training_error_source_layer_names;
		std::vector<std::string> training_exclude_data_update_layer_names;
		std::string training_algo;
		int training_epoch_count;
		float learning_rate;
		int learning_rate_decay_tail_epoch_count;
		int learning_rate_rise_head_epoch_count;
		float learning_rate_decay_rate;
		float learning_rate_rise_rate;
		float weight_decay;
		int batch_size;
		std::string momentum_type_str;
		float momentum_val;
		bool load_resume;
		bool dump_resume;
		int ann_count;
		int batch_offset;

		debug_state::ptr debug;

	protected:
		static const char * logfile_name;
		static const char * ann_subfolder_name;
		static const char * debug_subfolder_name;
		static const char * trained_ann_index_extractor_pattern;
		static const char * resume_ann_index_extractor_pattern;
		static const char * ann_resume_subfolder_name;

		std::string default_config_path;

	private:
		toolset();
		toolset(const toolset&);
	};
}
