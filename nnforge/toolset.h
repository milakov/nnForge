/*
 *  Copyright 2011-2016 Maxim Milakov
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
#include "stream_redirector.h"
#include "network_trainer.h"
#include "structured_data_stream_reader.h"
#include "data_transformer.h"
#include "normalize_data_transformer.h"

#include <vector>
#include <string>

namespace nnforge
{
	class toolset
	{
	public:
		toolset(factory_generator::ptr master_factory);

		virtual ~toolset() = default;

		// Returns true if action is specified
		bool parse(int argc, char* argv[]);

		std::string get_action() const;

		void do_action();

	protected:
		enum dataset_usage
		{
			dataset_usage_train = 0,
			dataset_usage_validate_when_train = 1,
			dataset_usage_inference = 2,
			dataset_usage_dump_data = 3,
			dataset_usage_create_normalizer = 4,
			dataset_usage_check_gradient = 5,
			dataset_usage_shuffle_data = 6,
			dataset_usage_update_bn_weights = 7
		};

		enum schema_usage
		{
			schema_usage_train = 0,
			schema_usage_validate_when_train = 1,
			schema_usage_inference = 2,
			schema_usage_dump_schema = 3
		};

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

		virtual boost::filesystem::path get_input_data_folder() const;

		virtual network_schema::ptr load_schema() const;

		virtual network_schema::ptr get_schema(schema_usage usage) const;

		virtual std::map<unsigned int, std::map<std::string, std::pair<layer_configuration_specific, std::vector<double> > > > run_inference();

		virtual void dump_schema_gv();

		virtual void train();

		virtual boost::filesystem::path get_ann_subfolder_name() const;

		virtual network_trainer::ptr get_network_trainer() const;

		virtual std::vector<network_data_pusher::ptr> get_validators_for_training(network_schema::const_ptr schema);

		virtual bool is_training_with_validation() const;

		virtual void prepare_training_data();

		virtual void prepare_testing_data();

		virtual void shuffle_data();

		virtual void dump_data();

		virtual void dump_data_visual(structured_data_bunch_reader::ptr dr);

		virtual void dump_data_csv(structured_data_bunch_reader::ptr dr);

		virtual void create_normalizer();

		virtual void check_gradient();

		virtual void save_random_weights();

		virtual void update_bn_weights();

		virtual structured_data_bunch_reader::ptr get_structured_data_bunch_reader(
			const std::string& dataset_name,
			dataset_usage usage,
			unsigned int multiple_epoch_count,
			unsigned int shuffle_block_size) const;

		virtual raw_data_reader::ptr get_raw_reader(
			const std::string& dataset_name,
			const std::string& layer_name,
			dataset_usage usage,
			std::shared_ptr<std::istream> in) const;

		virtual structured_data_reader::ptr get_structured_reader(
			const std::string& dataset_name,
			const std::string& layer_name,
			dataset_usage usage,
			std::shared_ptr<std::istream> in) const;

		virtual std::vector<unsigned int> get_dump_data_dimension_list(unsigned int original_dimension_count) const;

		virtual std::vector<data_transformer::ptr> get_data_transformer_list(
			const std::string& dataset_name,
			const std::string& layer_name,
			dataset_usage usage) const;

		virtual float get_dataset_value_data_value(
			const std::string& dataset_name,
			dataset_usage usage) const;

	protected:

		structured_data_reader::ptr apply_transformers(
			structured_data_reader::ptr original_reader,
			const std::vector<data_transformer::ptr>& data_transformer_list) const;

		// Returns empty smart pointer if no normalize_data_transformer exists for the layer specified
		normalize_data_transformer::ptr get_normalize_data_transformer(const std::string& layer_name) const;

	private:
		void dump_settings();

		std::vector<std::pair<unsigned int, boost::filesystem::path> > get_ann_data_index_and_folderpath_list() const;

		std::vector<network_data_peek_entry> get_snapshot_ann_list_entry_list() const;

		std::set<unsigned int> get_trained_ann_list() const;

		std::map<unsigned int, unsigned int> get_snapshot_ann_list(const std::set<unsigned int>& exclusion_ann_list) const;

		unsigned int get_starting_index_for_batch_training() const;

		static bool compare_entry(network_data_peek_entry i, network_data_peek_entry j);

		std::map<std::string, boost::filesystem::path> get_data_filenames(const std::string& dataset_name) const;

	protected:
		factory_generator::ptr master_factory;

		forward_propagation_factory::ptr forward_prop_factory;
		backward_propagation_factory::ptr backward_prop_factory;

		std::shared_ptr<stream_duplicator> out_to_log_duplicator;
		std::shared_ptr<stream_redirector> out_to_log_redirector;

	protected:
		std::string action;
		boost::filesystem::path config_file_path;
		boost::filesystem::path input_data_folder;
		boost::filesystem::path working_data_folder;
		std::string schema_filename;
		std::vector<std::string> inference_output_layer_names;
		std::vector<std::string> inference_force_data_layer_names;
		std::string inference_dataset_name;
		std::string training_dataset_name;
		std::string shuffle_dataset_name;
		std::string normalizer_dataset_name;
		int inference_ann_data_index;
		bool debug_mode;
		bool profile_mode;
		std::vector<std::string> training_output_layer_names;
		std::vector<std::string> training_error_source_layer_names;
		std::vector<std::string> training_exclude_data_update_layer_names;
		std::string training_algo;
		int training_epoch_count;
		float learning_rate;
		std::string learning_rate_policy;
		float learning_rate_decay_rate;
		int learning_rate_decay_start_epoch;
		std::string step_learning_rate_epochs_and_rates;
		float weight_decay;
		int batch_size;
		std::string momentum_type_str;
		float momentum_val;
		float momentum_val2;
		bool resume_from_snapshot;
		bool dump_snapshot;
		int keep_snapshots_frequency;
		int ann_count;
		int batch_offset;
		std::string inference_mode;
		std::string inference_output_dataset_name;
		std::string dump_dataset_name;
		std::string dump_layer_name;
		std::string normalizer_layer_name;
		int dump_data_sample_count;
		std::string dump_extension_image;
		std::string dump_extension_video;
		bool dump_data_rgb;
		int dump_data_scale;
		int dump_data_video_fps;
		int epoch_count_in_training_dataset;
		int epoch_count_in_validating_dataset;
		int dump_compact_samples;
		std::string log_mode;
		float training_mix_validating_ratio;
		std::string dump_format;
		int shuffle_block_size;
		std::string check_gradient_weights;
		int check_gradient_max_weights_per_set;
		float check_gradient_base_step;
		float check_gradient_relative_threshold_warning;
		float check_gradient_relative_threshold_error;

		debug_state::ptr debug;
		profile_state::ptr profile;
		learning_rate_decay_policy::ptr lr_policy;

	protected:
		static const char * logfile_name;
		static const char * ann_subfolder_name;
		static const char * debug_subfolder_name;
		static const char * profile_subfolder_name;
		static const char * trained_ann_index_extractor_pattern;
		static const char * snapshot_ann_index_extractor_pattern;
		static const char * ann_snapshot_subfolder_name;
		static const char * dataset_extractor_pattern;
		static const char * dump_data_subfolder_name;
		static const char * dataset_value_data_layer_name;

		std::string default_config_path;

	private:
		toolset();
		toolset(const toolset&);
	};
}
